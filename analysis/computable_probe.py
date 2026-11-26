#!/usr/bin/env python3
"""Does Computable's Vast.ai query miss machines that should be in its book?

Plain-language version
  Computable publishes an open GPU price index (github.com/getcomputable/gpu-index,
  Apache-2.0). One of its inputs is Vast.ai. Its collector asks Vast for the
  rentable on-demand offers of one GPU model, cheapest first, at most 50. If
  fewer than 50 come back it treats the answer as the complete list.

  My own recorder found that Vast's search can leave out machines that are live:
  ask for the same machine by its id and it answers. That was seen with a
  different query (signed in, rentable or not). This script tests whether it
  also happens with THEIR query. It is run by hand. It changes nothing.

What their code does (read on 26 Nov 2026 at commit 528b639, dated 24 Sep 2026)
  src/gpu_index/observatory/sources/vast.py
    - GET https://console.vast.ai/api/v0/bundles/?q=<JSON>, no API key, with
        {"gpu_name": {"eq": NAME}, "rentable": {"eq": true}, "type": "on-demand",
         "order": [["dph_total", "asc"]], "limit": 50}
    - only if 50 offers come back, the same query again with order "desc".
      If that is also 50, the book is flagged coverage_gap.
    - no verified filter in the query. Verification is checked afterwards.
  src/gpu_index/index/sources.py, parse_vast_offers
    - an offer is dropped if num_gpus is not a whole number from 1 to 16, if
      dph_total is missing, or if is_bid is set
    - price per GPU = dph_total / num_gpus (dph_total includes default storage)
  vast.py, rank_machines: one row per machine, its cheapest per-GPU offer
  src/gpu_index/index/composite.py, us_ca_verified_host
    - verification must be exactly "verified", and the text after the last
      comma of the location must be US or CA
  src/gpu_index/index/panel.py, vast_vwm_verified_us_ca_floor
    - weighted median of those per-GPU prices. Each machine's weight is the GPU
      count of its cheapest offer, not the size of the machine.
    - held out as "thin_book" unless there are at least 5 machines and 3 hosts
      (the B200 panel uses the same median with no such floors)

What one run does, for each GPU model
  a. their exact query, without a key (1 call, or 2 if the first returns 50)
  b. my own query for the same model with my key, through the vastai CLI. By
     default it asks for rentable offers only, with no limit of 50. That keeps
     the cost to a few dozen rows of my daily quota, which the recorder needs.
     --wide-any asks for every offer instead (hundreds of rows).
  c. the machines to check = machines in (b), machines in my archived snapshots
     of the last 24 hours, and machines in their result. Each one that is NOT
     in their result is asked for by id, using their filters plus the machine
     id, without a key. At most --max-lookups per model, most likely first.
  d. their exact query once more, then each candidate is asked for by id a
     second time. A machine counts as missed only if it is absent from their
     result both before and after, AND asking by id finds a rentable offer both
     times. That rules out machines that became rentable or got rented mid-run.
     Control: before any of this, one machine that IS in their result is asked
     for by id. If that returns nothing, asking by id does not work in this
     form and the run stops, so a broken lookup can never look like "not
     rentable".
  e. the report: how many missed machines pass their own filters, and their
     seat's median with and without them.

What this cannot show
  It does not see Computable's production data. It shows what their published
  query returns from this machine at this moment. It also does not test machines
  that ARE in their result but may be shown with a pricier offer than exists.
  One run proves nothing. Run it about six times across a day, then --report.

Files written (never the recorder's own files)
  raw/computable_probe/dt=<date>/run=<run id>/NNN_<label>.json.gz   every reply, byte for byte
  manifest/computable_probe_requests.jsonl                          one row per request
  manifest/computable_probe_runs.jsonl                              one row per run

Usage (on the droplet)
  python3 analysis/computable_probe.py --root /root/data
  python3 analysis/computable_probe.py --root /root/data --seats h100_sxm
  python3 analysis/computable_probe.py --root /root/data --report
"""

import argparse
import gzip
import hashlib
import json
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "analysis"))
sys.path.insert(0, str(REPO / "collectors" / "vast"))

import vast_data  # noqa: E402
from collect_vast import is_quota_error, is_rate_limited, run_vast_cli  # noqa: E402

PROBE_VERSION = "computable-probe-0.1.0"
THEIR_COMMIT = "528b63925b005dd151a5d8600fdb1ea7a37fa2bb"
THEIR_URL = "https://console.vast.ai/api/v0/bundles/"
THEIR_LIMIT = 50
# Their collector identifies itself as "gpu-index/1.0". This probe says who it really is.
USER_AGENT = "gpu-price-mix-probe/0.1 (+https://github.com/pradeepu191/gpu-price-mix)"
SPACING_SECONDS = 1.0          # gap between requests. Theirs is 0.75
ARCHIVE_HOURS = 24
MAX_LOOKUPS = 12               # per GPU model, per run

# Each seat is one Vast entry in one of their index panels.
SEATS = {
    "h100_sxm": {"chips": ["H100 SXM"], "min_machines": 5, "min_hosts": 3,
                 "config": "config/index_panel_h100_sxm.json"},
    "b200": {"chips": ["B200"], "min_machines": None, "min_hosts": None,
             "config": "config/index_panel_b200.json"},
    "h200_broad": {"chips": ["H200", "H200 NVL"], "min_machines": 5, "min_hosts": 3,
                   "config": "config/index_panel_h200_broad.json"},
}


# ---------------------------------------------------------------- their logic, reproduced

def their_query(gpu_name, order="asc", machine_id=None):
    """The query their collector sends. machine_id is my addition, for asking about one machine."""
    query = {
        "gpu_name": {"eq": gpu_name},
        "rentable": {"eq": True},
        "type": "on-demand",
        "order": [["dph_total", order]],
        "limit": THEIR_LIMIT,
    }
    if machine_id is not None:
        query["machine_id"] = {"eq": int(machine_id)}
    return query


def their_url(query):
    return THEIR_URL + "?q=" + urllib.parse.quote(json.dumps(query))


def parse_their_offers(offers, gpu_name):
    """Keep the offers their parser would keep. Returns (kept, number skipped)."""
    kept = []
    skipped = 0
    for offer in offers:
        count = offer.get("num_gpus")
        total = offer.get("dph_total")
        whole = isinstance(count, (int, float)) and not isinstance(count, bool) and float(count).is_integer()
        if not whole or not 1 <= int(count) <= 16 or total is None:
            skipped += 1
            continue
        if offer.get("is_bid"):
            skipped += 1
            continue
        if offer.get("gpu_name") != gpu_name:
            skipped += 1
            continue
        kept.append({
            "offer_id": offer.get("id"),
            "machine_id": offer.get("machine_id"),
            "host_id": offer.get("host_id"),
            "num_gpus": int(count),
            "dph_total": float(total),
            "per_gpu": float(total) / int(count),
            "geolocation": str(offer.get("geolocation") or "?"),
            "verification": str(offer.get("verification") or "?"),
            "gpu_name": gpu_name,
        })
    return kept, skipped


def rank_machines(candidates):
    """One row per machine: its cheapest per-GPU offer. Sorted cheapest first."""
    best = {}
    for candidate in candidates:
        machine_id = candidate["machine_id"]
        current = best.get(machine_id)
        if current is None or candidate["per_gpu"] < current["per_gpu"]:
            best[machine_id] = candidate
    return sorted(best.values(), key=lambda row: row["per_gpu"])


def is_verified_us_ca(verification, geolocation):
    """Their population screen: a verified host whose location ends in US or CA."""
    if verification != "verified":
        return False
    country = str(geolocation or "").rsplit(",", 1)[-1].strip().upper()
    return country in ("US", "CA")


def weighted_median(pairs):
    """Their weighted median over (value, weight) pairs.

    The value where the running weight first passes half the total. If the
    running weight lands exactly on half, the average of that value and the next.
    """
    ordered = sorted(pairs)
    half = sum(weight for _, weight in ordered) / 2.0
    running = 0.0
    for index, (value, weight) in enumerate(ordered):
        running += weight
        if running > half:
            return value
        if running == half:
            return (value + ordered[index + 1][0]) / 2.0
    raise ValueError("weighted median of an empty book")


def seat_statistic(machine_rows, min_machines=None, min_hosts=None):
    """What the seat would publish from these machine rows (one row per machine)."""
    eligible = []
    for row in machine_rows:
        if is_verified_us_ca(row["verification"], row["geolocation"]):
            eligible.append(row)
    machines = set()
    hosts = set()
    for row in eligible:
        if row.get("machine_id") is not None:
            machines.add(row["machine_id"])
        if row.get("host_id") is not None:
            hosts.add(row["host_id"])
    result = {"eligible_machines": len(machines), "eligible_hosts": len(hosts),
              "status": "priced", "median": None, "gpu_volume": 0}
    if not eligible:
        result["status"] = "no eligible machines"
        return result
    if min_machines is not None and (len(machines) < min_machines or len(hosts) < min_hosts):
        result["status"] = "held out (thin_book)"
        return result
    book = []
    for row in eligible:
        book.append((row["per_gpu"], row["num_gpus"]))
    result["median"] = round(weighted_median(book), 6)
    result["gpu_volume"] = sum(weight for _, weight in book)
    return result


def their_book(asc_offers, desc_offers, gpu_name):
    """Their candidate set for one GPU model, from the ascending reply and, if made, the descending one."""
    candidates, skipped = parse_their_offers(asc_offers, gpu_name)
    raw_count = len(asc_offers)
    truncated = raw_count >= THEIR_LIMIT
    coverage_gap = False
    if desc_offers is not None:
        more, more_skipped = parse_their_offers(desc_offers, gpu_name)
        seen = set()
        for candidate in candidates:
            seen.add(candidate["offer_id"])
        for candidate in more:
            if candidate["offer_id"] not in seen:
                candidates.append(candidate)
        skipped += more_skipped
        raw_count += len(desc_offers)
        coverage_gap = len(desc_offers) >= THEIR_LIMIT
    machines = {}
    for row in rank_machines(candidates):
        machines[str(row["machine_id"])] = row
    return {"raw_count": raw_count, "fetch_truncated": truncated, "coverage_gap": coverage_gap,
            "skipped": skipped, "machines": machines}


# ---------------------------------------------------------------- comparing

def known_machines(offers, gpu_name):
    """What my own data says about each machine of this GPU model: labels for machines not rentable now."""
    known = {}
    for offer in offers:
        if offer.get("gpu_name") != gpu_name or not offer.get("num_gpus"):
            continue
        machine_id = str(offer["machine_id"])
        entry = known.setdefault(machine_id, {"verification": None, "geolocation": None, "host_id": None,
                                              "rentable_seen": False, "cards": set(), "largest_offer": 0,
                                              "per_gpu": None})
        entry["verification"] = offer.get("verification")
        entry["geolocation"] = offer.get("geolocation")
        entry["host_id"] = offer.get("host_id")
        if offer.get("rentable"):
            entry["rentable_seen"] = True
        for gpu_id in offer.get("gpu_ids") or []:
            entry["cards"].add(gpu_id)
        entry["largest_offer"] = max(entry["largest_offer"], int(offer["num_gpus"]))
        if offer.get("dph_total") is not None:
            price = float(offer["dph_total"]) / int(offer["num_gpus"])
            if entry["per_gpu"] is None or price < entry["per_gpu"]:
                entry["per_gpu"] = price
    return known


def lookup_order(missing_ids, wide_known, archive_known):
    """Most informative first: rentable in my own query, then hidden from it, then rentable recently, then the rest."""
    def rank(machine_id):
        in_wide = machine_id in wide_known
        if in_wide and wide_known[machine_id]["rentable_seen"]:
            return 0
        if not in_wide and archive_known.get(machine_id, {}).get("rentable_seen"):
            return 1
        if not in_wide:
            return 2
        return 3
    return sorted(missing_ids, key=lambda machine_id: (rank(machine_id), machine_id))


def compare_chip(gpu_name, before, after, direct, wide_known, archive_known, looked_up, direct_again=None):
    """Label every machine that is missing from their result.

    before, after : their_book() results for the two runs of their query
    direct        : machine id -> offers returned when that machine was asked for by id with their filters
    looked_up     : the machine ids that were asked for by id
    direct_again  : the same question asked a second time, after their second run
    """
    direct_again = direct_again or {}
    union = set(before["machines"]) | set(wide_known) | set(archive_known)
    missing = sorted(union - set(before["machines"]))
    labels = []
    for machine_id in missing:
        info = wide_known.get(machine_id) or archive_known.get(machine_id) or {}
        label = {
            "machine_id": machine_id,
            "in_my_query": machine_id in wide_known,
            "in_archive": machine_id in archive_known,
            "looked_up": machine_id in looked_up,
            "rentable_now": None,
            "verified": info.get("verification") == "verified" if info else None,
            "us_ca": is_verified_us_ca("verified", info.get("geolocation")) if info else None,
            "gpu_count": (len(info["cards"]) or info["largest_offer"]) if info else None,
            "per_gpu": info.get("per_gpu"),
            "in_their_second_run": machine_id in after["machines"],
            "seen_once": False,
            "missed": False,
            "passes_their_filters": False,
            "row": None,
        }
        if machine_id in looked_up:
            kept, _ = parse_their_offers(direct.get(machine_id) or [], gpu_name)
            ranked = rank_machines(kept)
            label["rentable_now"] = bool(ranked)
            if ranked:
                row = ranked[0]
                label["verified"] = row["verification"] == "verified"
                label["us_ca"] = is_verified_us_ca("verified", row["geolocation"])
                label["per_gpu"] = row["per_gpu"]
                label["cheapest_offer_gpus"] = row["num_gpus"]
                label["row"] = row
                label["seen_once"] = not label["in_their_second_run"]
                again, _ = parse_their_offers(direct_again.get(machine_id) or [], gpu_name)
                label["missed"] = label["seen_once"] and bool(again)
                label["passes_their_filters"] = label["missed"] and is_verified_us_ca(
                    row["verification"], row["geolocation"])
        labels.append(label)
    return labels


def seat_report(seat, chip_results):
    """The seat's statistic from their result, and again with the missed machines added."""
    theirs = []
    added = []
    for chip in chip_results:
        theirs.extend(chip["before"]["machines"].values())
        for label in chip["labels"]:
            if label["missed"]:
                added.append(label["row"])
    without = seat_statistic(theirs, seat["min_machines"], seat["min_hosts"])
    with_missed = seat_statistic(theirs + added, seat["min_machines"], seat["min_hosts"])
    # "changed" means what the seat would publish changes: priced or held out, or the median itself.
    return {"as their query sees it": without, "with the missed machines": with_missed,
            "changed": (without["status"], without["median"]) != (with_missed["status"], with_missed["median"]),
            "counts_changed": without != with_missed}


# ---------------------------------------------------------------- requests and files

def anonymous_get(url):
    """One GET without a key. Returns (status code, body bytes, problem text or None)."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, response.read(), None
    except urllib.error.HTTPError as error:
        return error.code, error.read(), f"HTTP {error.code}"
    except Exception as error:  # a timeout or a dropped connection is recorded, not raised
        return None, b"", f"{type(error).__name__}: {error}"


class Recorder:
    """Saves every reply and writes one log row per request, in the recorder's style."""

    def __init__(self, root, run_id, run_date):
        self.root = Path(root)
        self.run_id = run_id
        self.raw_dir = self.root / "raw" / "computable_probe" / f"dt={run_date}" / f"run={run_id}"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        (self.root / "manifest").mkdir(parents=True, exist_ok=True)
        self.log_path = self.root / "manifest" / "computable_probe_requests.jsonl"
        self.calls = {"without a key": 0, "with my key": 0}
        self.key_rows = 0
        self.sequence = 0

    def save(self, label, transport, request_text, raw_bytes, status, n_offers, problem):
        self.sequence += 1
        self.calls[transport] += 1
        if transport == "with my key" and n_offers:
            self.key_rows += n_offers
        raw_path = self.raw_dir / f"{self.sequence:03d}_{label}.json.gz"
        with gzip.open(raw_path, "wb") as handle:
            handle.write(raw_bytes)
        row = {
            "run_id": self.run_id,
            "probe_version": PROBE_VERSION,
            "request": self.sequence,
            "label": label,
            "transport": transport,
            "query": request_text,
            "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "bytes": len(raw_bytes),
            "sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "raw_file": str(raw_path.relative_to(self.root)),
            "n_offers": n_offers,
            "problem": problem,
        }
        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")


class Stop(Exception):
    """Raised when Vast says to slow down or the daily quota is gone. The run ends at once."""


def ask_their_way(recorder, get, sleep, label, query):
    """Send one of their-style queries. Returns the list of offers."""
    url = their_url(query)
    sleep(SPACING_SECONDS)
    status, body, problem = get(url)
    offers = None
    if problem is None:
        try:
            offers = json.loads(body).get("offers") or []
        except (ValueError, AttributeError):
            problem = "reply was not the expected JSON"
    recorder.save(label, "without a key", json.dumps(query), body, status,
                  len(offers) if offers is not None else None, problem)
    if status == 429:
        raise Stop(f"Vast answered 429 (too many requests) on {label}")
    if offers is None:
        raise Stop(f"{label} failed: {problem}")
    return offers


def ask_with_key(recorder, cli, sleep, label, query):
    """Send one query through the vastai CLI with my key. Returns the list of offers, or None if it failed."""
    sleep(SPACING_SECONDS)
    exit_code, stdout, stderr = cli(query)
    offers = None
    problem = None
    try:
        parsed = json.loads(stdout)
    except ValueError:
        parsed = None
        problem = f"reply was not JSON (exit code {exit_code}): {stderr[:200]}"
    if isinstance(parsed, list):
        offers = parsed
    elif parsed is not None:
        problem = f"error reply: {str(parsed)[:200]}"
    recorder.save(label, "with my key", query, stdout, exit_code,
                  len(offers) if offers is not None else None, problem)
    if is_quota_error(parsed):
        raise Stop("my daily Vast quota is used up. The recorder needs it more than this probe does")
    if is_rate_limited(parsed):
        raise Stop("Vast answered 429 (too many requests) to my key")
    return offers


def archive_offers(root, now, hours=ARCHIVE_HOURS):
    """Every offer in my complete on-demand snapshots of the last `hours` hours. Read only."""
    earliest = (now - timedelta(hours=hours)).strftime("%Y%m%dT%H%M%SZ")
    offers = []
    try:
        run_ids = vast_data.complete_runs(root, "on-demand")
    except FileNotFoundError:
        return offers, 0
    recent = []
    for run_id in run_ids:
        if run_id >= earliest:
            recent.append(run_id)
    for run_id in recent:
        offers.extend(vast_data.load_offers(root, "on-demand", run_id))
    return offers, len(recent)


def inside_recorder_window(now):
    """The recorder starts on even UTC hours and takes a few minutes. Keep clear of it."""
    return now.hour % 2 == 0 and now.minute < 15


def probe_chip(recorder, get, cli, sleep, gpu_name, archived, max_lookups, wide_any):
    slug = gpu_name.replace(" ", "_")

    def their_two_calls(tag):
        asc = ask_their_way(recorder, get, sleep, f"{slug}_theirs_{tag}_asc", their_query(gpu_name, "asc"))
        desc = None
        if len(asc) >= THEIR_LIMIT:
            desc = ask_their_way(recorder, get, sleep, f"{slug}_theirs_{tag}_desc", their_query(gpu_name, "desc"))
        return their_book(asc, desc, gpu_name)

    before = their_two_calls("before")

    # Control: a machine their query DID return must also answer when asked for by id.
    if before["machines"]:
        control_id = sorted(before["machines"])[0]
        control = ask_their_way(recorder, get, sleep, f"{slug}_control_{control_id}",
                                their_query(gpu_name, "asc", control_id))
        kept, _ = parse_their_offers(control, gpu_name)
        if not kept:
            raise Stop(f"control failed for {gpu_name}: a machine in their result did not answer when asked "
                       f"for by id, so asking by id cannot be trusted in this form")

    rentable = "any" if wide_any else "true"
    mine = ask_with_key(recorder, cli, sleep, f"{slug}_mine", f"rentable={rentable} gpu_name={slug}")
    wide_known = known_machines(mine or [], gpu_name)
    archive_known = known_machines(archived, gpu_name)

    union = set(before["machines"]) | set(wide_known) | set(archive_known)
    missing = union - set(before["machines"])
    ordered = lookup_order(missing, wide_known, archive_known)
    direct = {}
    for machine_id in ordered[:max_lookups]:
        direct[machine_id] = ask_their_way(recorder, get, sleep, f"{slug}_machine_{machine_id}",
                                           their_query(gpu_name, "asc", machine_id))

    after = their_two_calls("after")
    direct_again = {}
    for machine_id, offers in direct.items():
        kept, _ = parse_their_offers(offers, gpu_name)
        if kept and machine_id not in after["machines"]:
            direct_again[machine_id] = ask_their_way(recorder, get, sleep, f"{slug}_again_{machine_id}",
                                                     their_query(gpu_name, "asc", machine_id))
    labels = compare_chip(gpu_name, before, after, direct, wide_known, archive_known, set(direct), direct_again)
    return {"gpu_name": gpu_name, "before": before, "after": after, "labels": labels,
            "my_query_worked": mine is not None, "my_query_offers": len(mine or []),
            "machines_in_my_query": len(wide_known), "machines_in_archive": len(archive_known)}


def chip_summary(chip):
    """Counts for the run log. Machine ids of missed machines are kept so a run can be re-checked."""
    labels = chip["labels"]
    summary = {
        "gpu_name": chip["gpu_name"],
        "their_offers": chip["before"]["raw_count"],
        "their_machines": len(chip["before"]["machines"]),
        "their_fetch_truncated": chip["before"]["fetch_truncated"],
        "their_coverage_gap": chip["before"]["coverage_gap"],
        "my_query_worked": chip["my_query_worked"],
        "my_query_offers": chip["my_query_offers"],
        "machines_in_my_query": chip["machines_in_my_query"],
        "machines_in_archive": chip["machines_in_archive"],
        "missing_from_theirs": len(labels),
        "looked_up": sum(1 for label in labels if label["looked_up"]),
        "not_looked_up": sum(1 for label in labels if not label["looked_up"]),
        "not_rentable_now": sum(1 for label in labels if label["rentable_now"] is False),
        "changed_during_run": sum(1 for label in labels if label["rentable_now"] and not label["missed"]),
        "missed_rentable": sum(1 for label in labels if label["missed"]),
        "missed_passing_their_filters": sum(1 for label in labels if label["passes_their_filters"]),
        "missed_machine_ids": [label["machine_id"] for label in labels if label["missed"]],
    }
    return summary


def run_probe(root, seat_names, get, cli, now, sleep=time.sleep, max_lookups=MAX_LOOKUPS, wide_any=False):
    """One full run. Returns the run row that was written to the run log."""
    run_id = now.strftime("%Y%m%dT%H%M%SZ")
    recorder = Recorder(root, run_id, now.strftime("%Y-%m-%d"))
    archived, archive_runs = archive_offers(root, now)
    run = {"run_id": run_id, "probe_version": PROBE_VERSION, "their_commit": THEIR_COMMIT,
           "started_at_utc": now.isoformat(), "seats": {}, "chips": [], "stopped": None,
           "archive_snapshots_used": archive_runs, "max_lookups": max_lookups, "wide_any": wide_any}
    results = {}
    try:
        for seat_name in seat_names:
            for gpu_name in SEATS[seat_name]["chips"]:
                if gpu_name not in results:
                    results[gpu_name] = probe_chip(recorder, get, cli, sleep, gpu_name, archived,
                                                   max_lookups, wide_any)
    except Stop as stop:
        run["stopped"] = str(stop)
    for seat_name in seat_names:
        chips = [results[name] for name in SEATS[seat_name]["chips"] if name in results]
        if len(chips) == len(SEATS[seat_name]["chips"]):
            run["seats"][seat_name] = seat_report(SEATS[seat_name], chips)
    for chip in results.values():
        run["chips"].append(chip_summary(chip))
    run["calls"] = dict(recorder.calls)
    run["calls_total"] = sum(recorder.calls.values())
    run["key_rows_used"] = recorder.key_rows
    run["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    with open(Path(root) / "manifest" / "computable_probe_runs.jsonl", "a", encoding="utf-8") as handle:
        handle.write(json.dumps(run) + "\n")
    run["details"] = results
    return run


# ---------------------------------------------------------------- printing

def describe_statistic(statistic):
    text = f"{statistic['eligible_machines']} verified US/CA machines on {statistic['eligible_hosts']} hosts, "
    if statistic["median"] is None:
        return text + statistic["status"]
    return text + f"median ${statistic['median']:.4f} per GPU-hour over {statistic['gpu_volume']} GPUs"


def print_run(run):
    print(f"\nrun {run['run_id']}   Vast calls: {run['calls_total']} "
          f"({run['calls']['without a key']} without a key, {run['calls']['with my key']} with my key, "
          f"{run['key_rows_used']} rows of my daily quota)")
    if run["stopped"]:
        print(f"  STOPPED EARLY: {run['stopped']}")
    for chip in run["chips"]:
        print(f"\n  {chip['gpu_name']}")
        flags = ""
        if chip["their_fetch_truncated"]:
            flags = "  (hit the limit of 50, descending call made)"
        if chip["their_coverage_gap"]:
            flags = "  (both calls full: their code flags coverage_gap)"
        print(f"    their query returned {chip['their_offers']} offers on {chip['their_machines']} machines{flags}")
        if chip["my_query_worked"]:
            print(f"    my query returned {chip['my_query_offers']} offers on {chip['machines_in_my_query']} machines")
        else:
            print("    my query failed, so only the archive and their result were used")
        print(f"    machines known to me but not in their result: {chip['missing_from_theirs']}  "
              f"(asked by id: {chip['looked_up']}, not asked: {chip['not_looked_up']})")
        print(f"      not rentable right now:                                  {chip['not_rentable_now']}")
        print(f"      rentable by id but changed during the run (not counted): {chip['changed_during_run']}")
        print(f"      MISSED: rentable by id twice, absent before and after:   {chip['missed_rentable']}")
        print(f"        of those, verified and in the US or Canada:            {chip['missed_passing_their_filters']}")
        details = run.get("details", {}).get(chip["gpu_name"])
        if details:
            for label in details["labels"]:
                if label["missed"]:
                    print(f"        machine {label['machine_id']}: ${label['per_gpu']:.4f} per GPU-hour, "
                          f"cheapest offer has {label['cheapest_offer_gpus']} GPUs, "
                          f"verified {label['verified']}, US/CA {label['us_ca']}, "
                          f"in my query {label['in_my_query']}")
    for seat_name, report in run["seats"].items():
        print(f"\n  seat {seat_name} ({SEATS[seat_name]['config']})")
        print(f"    as their query sees it:    {describe_statistic(report['as their query sees it'])}")
        print(f"    with the missed machines:  {describe_statistic(report['with the missed machines'])}")
        print(f"    would the seat's published result change: {'YES' if report['changed'] else 'no'}"
              + ("  (machine and host counts change, the result does not)"
                 if report["counts_changed"] and not report["changed"] else ""))
    print("\nOne run proves nothing. Run about six times across a day, then use --report.")


def print_report(root):
    path = Path(root) / "manifest" / "computable_probe_runs.jsonl"
    if not path.exists():
        sys.exit("No probe runs yet.")
    runs = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            runs.append(json.loads(line))
    print(f"{len(runs)} probe runs\n")
    print("run_id            model      their machines  asked by id  missed  missed passing their filters")
    for run in runs:
        for chip in run["chips"]:
            print(f"{run['run_id']}  {chip['gpu_name']:9}  {chip['their_machines']:14d}  {chip['looked_up']:11d}  "
                  f"{chip['missed_rentable']:6d}  {chip['missed_passing_their_filters']:28d}")
        if run["stopped"]:
            print(f"{run['run_id']}  stopped early: {run['stopped']}")
    print("\nseat results, run by run")
    for run in runs:
        for seat_name, report in run["seats"].items():
            mark = "CHANGED" if report["changed"] else ("counts" if report.get("counts_changed") else "same")
            print(f"{run['run_id']}  {seat_name:11}  {mark:7}  theirs: {describe_statistic(report['as their query sees it'])}")
            if report["changed"] or report.get("counts_changed"):
                print(f"{'':18}{'':11}  {'':7}  with:   {describe_statistic(report['with the missed machines'])}")
    with_misses = sum(1 for run in runs if any(chip["missed_rentable"] for chip in run["chips"]))
    with_eligible = sum(1 for run in runs if any(chip["missed_passing_their_filters"] for chip in run["chips"]))
    changed = sum(1 for run in runs if any(report["changed"] for report in run["seats"].values()))
    print(f"\nruns with any missed rentable machine: {with_misses} of {len(runs)}")
    print(f"runs with a missed machine that passes their filters: {with_eligible} of {len(runs)}")
    print(f"runs where a seat's published result would change: {changed} of {len(runs)}")
    print("(CHANGED = priced or held out flips, or the median moves. counts = only the machine or host counts move.)")


def main():
    parser = argparse.ArgumentParser(description="Does Computable's Vast query miss machines?")
    parser.add_argument("--root", type=Path, default=Path("/root/data"))
    parser.add_argument("--seats", default="h100_sxm,b200,h200_broad",
                        help=f"comma-separated, from: {', '.join(SEATS)}")
    parser.add_argument("--max-lookups", type=int, default=MAX_LOOKUPS, help="machines asked by id, per GPU model")
    parser.add_argument("--wide-any", action="store_true",
                        help="my query asks for every offer, not just rentable ones (costs hundreds of quota rows)")
    parser.add_argument("--vastai", default=None, help="path to the vastai CLI")
    parser.add_argument("--force", action="store_true", help="run even while the recorder is likely running")
    parser.add_argument("--report", action="store_true", help="summarize all probe runs so far and exit")
    args = parser.parse_args()

    if args.report:
        print_report(args.root)
        return

    seat_names = [name.strip() for name in args.seats.split(",") if name.strip()]
    for name in seat_names:
        if name not in SEATS:
            sys.exit(f"Unknown seat {name!r}. Choose from: {', '.join(SEATS)}")
    now = datetime.now(timezone.utc)
    if inside_recorder_window(now) and not args.force:
        sys.exit("The recorder runs at the start of even UTC hours. Wait until quarter past, or use --force.")
    vastai_path = args.vastai or shutil.which("vastai")
    if not vastai_path:
        sys.exit("vastai CLI not found. Activate the venv or pass --vastai.")

    def cli(query):
        return run_vast_cli(vastai_path, "on-demand", query)

    run = run_probe(args.root, seat_names, anonymous_get, cli, now,
                    max_lookups=args.max_lookups, wide_any=args.wide_any)
    print_run(run)


if __name__ == "__main__":
    main()

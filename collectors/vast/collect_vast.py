#!/usr/bin/env python3
"""Hourly capture of the full Vast.ai offer market. Capture only, no analysis.

Scope: datacenter GPU models only, the hardware that GPU price indices cover.
Vast enforces a daily quota on search rows, so the whole market (thousands of
consumer cards) cannot be captured every hour, and is not needed.

Why price bands: Vast caps every search at 512 results. The server's sort order
cannot be trusted to give the cheapest 512 (observed for on-demand in Nov 2026),
so the collector never relies on ordering. Each model is queried once. If that
query returns 512 offers it may be truncated, so it is split into price bands,
from a lower price up to but not including an upper price, and each band is
queried again. A query that returns fewer than 512 offers is complete.

Each run, for each market type:
  - archive every query's raw output, gzipped, byte for byte
  - append one manifest row per query, including failed and truncated ones
  - append one summary row per market type, saying whether the read was complete

Usage:
  python3 collect_vast.py --root /root/data
Environment:
  HEALTHCHECK_URL   optional. Pinged after a clean run, /fail is pinged otherwise.
"""

import argparse
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

COLLECTOR_VERSION = "vast-0.3.0"
PAGE_CAP = 512              # observed server-side cap per search, Nov 2026
MAX_QUERIES = 200           # safety stop per market type
PRICE_DECIMALS = 6          # band edges are rounded to this many decimals
BID_EVERY_HOURS = 6         # the interruptible market is captured less often, to save quota

# Exact names as Vast spells them (public pricing feed, Nov 2026).
# The search syntax wants underscores in place of spaces.
GPU_MODELS = [
    "A100 PCIE", "A100 SXM4",
    "H100 PCIE", "H100 SXM", "H100 NVL",
    "H200", "H200 NVL",
    "B200", "B300", "GB200",
    "L40S",
]


def utc_now():
    return datetime.now(timezone.utc)


def split_band(lower, upper):
    """Split a band that may be truncated. Returns None if it is too narrow to split."""
    if lower is None:
        lower = 0.0
    if upper is None:
        # Open-ended top band: split at double its lower edge
        middle = round(max(lower * 2, lower + 1.0), PRICE_DECIMALS)
        return [(lower, middle), (middle, None)]
    middle = round((lower + upper) / 2, PRICE_DECIMALS)
    if middle <= lower or middle >= upper:
        return None
    return [(lower, middle), (middle, upper)]


def build_query(model, lower, upper):
    """Vast query for one GPU model, optionally inside a price band.
    rentable=any includes free and occupied machines."""
    query = "rentable=any gpu_name=" + model.replace(" ", "_")
    if lower is not None:
        query += f" dph_total>={lower:.{PRICE_DECIMALS}f}"
    if upper is not None:
        query += f" dph_total<{upper:.{PRICE_DECIMALS}f}"
    return query


def is_quota_error(parsed):
    """Vast answers an exhausted daily quota with a JSON object carrying status_code 429."""
    return isinstance(parsed, dict) and parsed.get("status_code") == 429


def run_vast_cli(vastai_path, market_type, query):
    """Run one search through the official CLI. Returns (exit_code, stdout_bytes, stderr_text)."""
    command = [
        vastai_path, "search", "offers", query,
        "--no-default",
        "--type", market_type,
        "--limit", "100000",        # the server applies its own cap
        "--raw",
    ]
    completed = subprocess.run(command, capture_output=True, timeout=180)
    return completed.returncode, completed.stdout, completed.stderr.decode("utf-8", "replace")


def read_market(run_query, market_type, models=GPU_MODELS, page_cap=PAGE_CAP):
    """Read one market type, model by model. Pure logic, so it can be tested with a fake run_query.

    run_query(market_type, query) must return (exit_code, stdout_bytes, stderr_text).
    Returns (pages, summary). Each page is one query's raw bytes plus diagnostics.
    """
    pages = []
    seen_ids = set()
    rows_returned = 0
    status = "complete"

    # Work list of (model, lower, upper). No band at first: one query per model.
    work = []
    for model in models:
        work.append((model, None, None))

    while work:
        if len(pages) >= MAX_QUERIES:
            status = "incomplete"
            break

        model, lower, upper = work.pop(0)
        query = build_query(model, lower, upper)
        fetched_at = utc_now().isoformat()
        exit_code, stdout, stderr = run_query(market_type, query)

        page = {
            "page": len(pages), "query": query, "model": model,
            "band_lower": lower, "band_upper": upper,
            "fetched_at_utc": fetched_at, "exit_code": exit_code, "raw": stdout,
            "stderr": stderr[:500], "n_offers": None, "n_new": None,
            "capped": None, "problem": None,
        }
        pages.append(page)

        # A failed or unreadable query ends this market's read, and the failure is recorded
        if exit_code != 0:
            page["problem"] = "cli_error"
            status = "failed"
            break
        try:
            offers = json.loads(stdout)
        except json.JSONDecodeError:
            page["problem"] = "not_json"
            status = "failed"
            break
        if is_quota_error(offers):
            # Stop immediately. Further queries would only fail too.
            page["problem"] = "quota_exhausted"
            status = "quota_exhausted"
            break
        if not isinstance(offers, list):
            page["problem"] = "not_a_list"
            status = "failed"
            break

        # Keep every offer we see, even from a truncated band. Ids remove repeats.
        new_count = 0
        for offer in offers:
            if offer["id"] not in seen_ids:
                seen_ids.add(offer["id"])
                new_count += 1
        page["n_offers"] = len(offers)
        page["n_new"] = new_count
        rows_returned += len(offers)
        page["capped"] = len(offers) >= page_cap

        # Fewer than the cap means this band is complete
        if not page["capped"]:
            continue

        # At the cap the band may be truncated, so split it and query both halves
        halves = split_band(lower, upper)
        if halves is None:
            # 512 or more offers inside a band narrower than a millionth of a dollar
            page["problem"] = "tie_overflow"
            status = "incomplete"
            continue
        for half_lower, half_upper in reversed(halves):
            work.insert(0, (model, half_lower, half_upper))

    summary = {
        "market_type": market_type,
        "status": status,
        "queries": len(pages),
        "rows_returned": rows_returned,
        "unique_offers": len(seen_ids),
    }
    return pages, summary


def archive_market(root, run_id, run_date, market_type, pages, summary):
    """Write raw pages and append manifest rows. Called even when the read failed."""
    raw_dir = root / "raw" / "vast" / f"type={market_type}" / f"dt={run_date}" / f"run={run_id}"
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = root / "manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    with open(manifest_dir / "vast_pages.jsonl", "a", encoding="utf-8") as page_log:
        for page in pages:
            raw_bytes = page["raw"]
            raw_path = raw_dir / f"page_{page['page']:03d}.json.gz"
            with gzip.open(raw_path, "wb") as handle:
                handle.write(raw_bytes)
            row = {
                "run_id": run_id,
                "collector_version": COLLECTOR_VERSION,
                "market_type": market_type,
                "page": page["page"],
                "query": page["query"],
                "model": page["model"],
                "band_lower": page["band_lower"],
                "band_upper": page["band_upper"],
                "fetched_at_utc": page["fetched_at_utc"],
                "exit_code": page["exit_code"],
                "bytes": len(raw_bytes),
                "sha256": hashlib.sha256(raw_bytes).hexdigest(),
                "raw_file": str(raw_path.relative_to(root)),
                "n_offers": page["n_offers"],
                "n_new": page["n_new"],
                "capped": page["capped"],
                "problem": page["problem"],
                "stderr": page["stderr"],
            }
            page_log.write(json.dumps(row) + "\n")

    with open(manifest_dir / "vast_runs.jsonl", "a", encoding="utf-8") as run_log:
        row = dict(summary)
        row["run_id"] = run_id
        row["collector_version"] = COLLECTOR_VERSION
        row["finished_at_utc"] = utc_now().isoformat()
        run_log.write(json.dumps(row) + "\n")


def ping_healthcheck(all_ok):
    """Tell an external monitor the run happened, so a dead collector gets noticed."""
    url = os.environ.get("HEALTHCHECK_URL")
    if not url:
        return
    if not all_ok:
        url = url.rstrip("/") + "/fail"
    try:
        urllib.request.urlopen(url, timeout=10)
    except OSError as error:
        print(f"healthcheck ping failed: {error}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Hourly Vast.ai offer snapshot")
    parser.add_argument("--root", type=Path, default=Path("data"), help="data directory, kept out of git")
    parser.add_argument("--vastai", default=shutil.which("vastai") or "vastai", help="path to the vastai CLI")
    parser.add_argument("--force-bid", action="store_true", help="also capture the bid market this run")
    args = parser.parse_args()

    started = utc_now()
    run_id = started.strftime("%Y%m%dT%H%M%SZ")
    run_date = started.strftime("%Y-%m-%d")
    root = args.root.resolve()

    def run_query(market_type, query):
        return run_vast_cli(args.vastai, market_type, query)

    # On-demand every run. The interruptible (bid) market only every few hours.
    market_types = ["on-demand"]
    if started.hour % BID_EVERY_HOURS == 0 or args.force_bid:
        market_types.append("bid")

    all_ok = True
    for market_type in market_types:
        try:
            pages, summary = read_market(run_query, market_type)
        except Exception as error:
            # Record the crash as a run row, so a gap in the data is explained
            pages = []
            summary = {"market_type": market_type, "status": "crashed",
                       "queries": 0, "unique_offers": 0, "error": repr(error)[:500]}
        archive_market(root, run_id, run_date, market_type, pages, summary)
        print(json.dumps(summary))
        if summary["status"] != "complete":
            all_ok = False
        if summary["status"] == "quota_exhausted":
            break

    ping_healthcheck(all_ok)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

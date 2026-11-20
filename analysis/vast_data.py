"""Load Vast offers from the raw archive the collector writes.

Only reads files. Never calls the Vast API, so it uses no quota.
"""

import gzip
import json
from pathlib import Path


def latest_complete_run(root, market_type):
    """Return the run_id of the newest run whose read of this market was complete."""
    runs_file = Path(root) / "manifest" / "vast_runs.jsonl"
    latest = None
    with open(runs_file, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["market_type"] != market_type:
                continue
            if row["status"] != "complete":
                continue
            # run ids are UTC timestamps like 20261119T040418Z, so text order is time order
            if latest is None or row["run_id"] > latest:
                latest = row["run_id"]
    if latest is None:
        raise FileNotFoundError(f"No complete {market_type} run in {runs_file}")
    return latest


def complete_runs(root, market_type):
    """Every run_id whose read of this market was complete, oldest first."""
    runs_file = Path(root) / "manifest" / "vast_runs.jsonl"
    run_ids = []
    with open(runs_file, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["market_type"] == market_type and row["status"] == "complete":
                run_ids.append(row["run_id"])
    return sorted(set(run_ids))


def load_offers(root, market_type, run_id):
    """Read every raw page of one run and return its offers, each offer once."""
    pattern = f"raw/vast/type={market_type}/dt=*/run={run_id}/page_*.json.gz"
    page_files = sorted(Path(root).glob(pattern))
    if not page_files:
        raise FileNotFoundError(f"No raw pages match {pattern}")

    offers_by_id = {}
    for page_file in page_files:
        with gzip.open(page_file, "rb") as handle:
            page = json.loads(handle.read())
        if not isinstance(page, list):
            continue   # an error reply saved as evidence, not offers
        for offer in page:
            offers_by_id[offer["id"]] = offer
    return list(offers_by_id.values())


def price_per_gpu_hour(offer):
    """GPU price only, in dollars per GPU per hour.

    dph_base is the GPU charge for the whole offer. Storage and bandwidth are
    billed separately and left out, so prices compare like with like.
    """
    return offer["dph_base"] / offer["num_gpus"]


def physical_gpus(offers, model=None):
    """One record per physical GPU card.

    Vast lists the same machine several times at once, as overlapping bundles
    (for example 1, 2, 4 and 8 GPUs of one 8-GPU machine), so counting offers
    counts the same cards repeatedly. gpu_ids names the actual cards, so each
    card is kept once.

    A card is free if any offer containing it can be rented right now. An
    offer shows as not rentable as soon as any one of its cards is taken, so
    "not rentable" on a big bundle does not mean all its cards are rented.
    Occupied here means: in no rentable offer. Our working assumption is that
    this means rented, which repeated snapshots let us check.
    """
    by_gpu = {}
    for offer in offers:
        if model is not None and offer["gpu_name"] != model:
            continue
        if not offer.get("num_gpus") or offer.get("dph_base") is None:
            continue
        price = price_per_gpu_hour(offer)
        for gpu_id in offer.get("gpu_ids") or []:
            record = by_gpu.get(gpu_id)
            if record is None:
                record = {
                    "gpu_id": gpu_id,
                    "gpu_name": offer["gpu_name"],
                    "machine_id": offer["machine_id"],
                    "host_id": offer["host_id"],
                    "price": price,
                    "free": False,
                }
                by_gpu[gpu_id] = record
            # Bundles of one machine nearly always share a per-GPU price. Keep the lowest if not.
            if price < record["price"]:
                record["price"] = price
            if offer["rentable"] is True:
                record["free"] = True

    cards = list(by_gpu.values())
    for card in cards:
        card["occupied"] = not card["free"]
    return cards


def machines(cards):
    """One record per machine: how many of the given cards it has, and their typical price."""
    by_machine = {}
    for card in cards:
        machine = by_machine.setdefault(card["machine_id"], {
            "machine_id": card["machine_id"], "host_id": card["host_id"], "prices": [], "gpus": 0})
        machine["prices"].append(card["price"])
        machine["gpus"] += 1
    result = []
    for machine in by_machine.values():
        prices = sorted(machine["prices"])
        machine["price"] = prices[len(prices) // 2]
        del machine["prices"]
        result.append(machine)
    return result

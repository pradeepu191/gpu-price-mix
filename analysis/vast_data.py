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
        for offer in page:
            offers_by_id[offer["id"]] = offer
    return list(offers_by_id.values())


def price_per_gpu_hour(offer):
    """GPU price only, in dollars per GPU per hour.

    dph_base is the GPU charge for the whole offer. Storage and bandwidth are
    billed separately and left out, so prices compare like with like.
    """
    return offer["dph_base"] / offer["num_gpus"]


def is_occupied(offer):
    """Working assumption, still to be confirmed over time: an offer Vast lists
    as not rentable is a machine someone is currently renting."""
    return offer["rentable"] is False

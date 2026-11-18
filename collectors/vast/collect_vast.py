#!/usr/bin/env python3
"""Hourly capture of the full Vast.ai offer market. Capture only, no analysis.

Why paging: Vast caps every search at 512 results, and the market is larger.
So each market type is read in ascending price order, one page at a time.
Each page asks for offers priced at or above the most expensive offer on the
previous page. Overlapping offers at the boundary are removed by offer id.

Each run, for each market type (on-demand and bid):
  - archive every page's raw output, gzipped, byte for byte
  - append one manifest row per page, including failed pages
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
import math
import os
import shutil
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

COLLECTOR_VERSION = "vast-0.1.0"
PAGE_CAP = 512              # observed server-side cap per search, Nov 2026
MAX_PAGES = 200             # safety stop, far above what the market needs
MARKET_TYPES = ["on-demand", "bid"]
PRICE_DECIMALS = 6          # boundary prices are floored to this many decimals


def utc_now():
    return datetime.now(timezone.utc)


def floor_price(price):
    """Round a price down, so the next page's lower bound never skips the boundary offer."""
    scale = 10 ** PRICE_DECIMALS
    return math.floor(price * scale) / scale


def build_query(lower_bound):
    """Vast query string. rentable=any means free and occupied machines both appear."""
    if lower_bound is None:
        return "rentable=any"
    return f"rentable=any dph_total>={lower_bound:.{PRICE_DECIMALS}f}"


def run_vast_cli(vastai_path, market_type, query):
    """Run one search through the official CLI. Returns (exit_code, stdout_bytes, stderr_text)."""
    command = [
        vastai_path, "search", "offers", query,
        "--no-default",
        "--type", market_type,
        "-o", "dph_total",          # ascending price, which the paging depends on
        "--limit", "100000",        # the server applies its own cap
        "--raw",
    ]
    completed = subprocess.run(command, capture_output=True, timeout=180)
    return completed.returncode, completed.stdout, completed.stderr.decode("utf-8", "replace")


def is_sorted_by_price(offers):
    """Paging is only correct if the server really returned offers in ascending price order."""
    for index in range(1, len(offers)):
        if offers[index]["dph_total"] < offers[index - 1]["dph_total"]:
            return False
    return True


def read_market(run_query, market_type, page_cap=PAGE_CAP):
    """Page through one market type. Pure logic, so it can be tested with a fake run_query.

    run_query(market_type, query) must return (exit_code, stdout_bytes, stderr_text).
    Returns (pages, summary). Each page is a dict holding its raw bytes and diagnostics.
    """
    pages = []
    seen_ids = set()
    lower_bound = None
    status = "complete"

    for page_number in range(MAX_PAGES):
        query = build_query(lower_bound)
        fetched_at = utc_now().isoformat()
        exit_code, stdout, stderr = run_query(market_type, query)

        page = {
            "page": page_number, "query": query, "fetched_at_utc": fetched_at,
            "exit_code": exit_code, "raw": stdout, "stderr": stderr[:500],
            "n_offers": None, "n_new": None, "problem": None,
        }
        pages.append(page)

        # A failed or unreadable page ends this market's read, and the failure is recorded
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
        if not isinstance(offers, list):
            page["problem"] = "not_a_list"
            status = "failed"
            break

        # Count offers this page adds, after removing boundary overlap
        new_count = 0
        for offer in offers:
            if offer["id"] not in seen_ids:
                seen_ids.add(offer["id"])
                new_count += 1
        page["n_offers"] = len(offers)
        page["n_new"] = new_count

        if not is_sorted_by_price(offers):
            page["problem"] = "unsorted"
            status = "incomplete"
            break

        # A page below the cap means we reached the most expensive offer
        if len(offers) < page_cap:
            break

        # Full page: continue from the highest price seen
        next_bound = floor_price(offers[-1]["dph_total"])
        if lower_bound is not None and next_bound <= lower_bound:
            # 512 or more offers share one price, so price paging cannot move forward
            page["problem"] = "tie_overflow"
            status = "incomplete"
            break
        lower_bound = next_bound
    else:
        status = "incomplete"   # hit MAX_PAGES without reaching the end

    summary = {
        "market_type": market_type,
        "status": status,
        "pages": len(pages),
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
                "fetched_at_utc": page["fetched_at_utc"],
                "exit_code": page["exit_code"],
                "bytes": len(raw_bytes),
                "sha256": hashlib.sha256(raw_bytes).hexdigest(),
                "raw_file": str(raw_path.relative_to(root)),
                "n_offers": page["n_offers"],
                "n_new": page["n_new"],
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
    args = parser.parse_args()

    started = utc_now()
    run_id = started.strftime("%Y%m%dT%H%M%SZ")
    run_date = started.strftime("%Y-%m-%d")
    root = args.root.resolve()

    def run_query(market_type, query):
        return run_vast_cli(args.vastai, market_type, query)

    all_ok = True
    for market_type in MARKET_TYPES:
        try:
            pages, summary = read_market(run_query, market_type)
        except Exception as error:
            # Record the crash as a run row, so a gap in the data is explained
            pages = []
            summary = {"market_type": market_type, "status": "crashed",
                       "pages": 0, "unique_offers": 0, "error": repr(error)[:500]}
        archive_market(root, run_id, run_date, market_type, pages, summary)
        print(json.dumps(summary))
        if summary["status"] != "complete":
            all_ok = False

    ping_healthcheck(all_ok)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

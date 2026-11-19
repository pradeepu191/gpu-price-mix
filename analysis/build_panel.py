#!/usr/bin/env python3
"""Flatten every complete Vast snapshot into one table, one row per offer per snapshot.

This is the "panel": the same offers seen again and again over time, which is
what lets us tell real price changes apart from changes in what is listed.
Reads only archived files, so it uses no Vast quota.

Columns kept are the ones the analysis needs. Network addresses and host
names are left out on purpose.

Usage:
  python3 analysis/build_panel.py --root /root/data --out /root/data/derived/vast_panel.csv
"""

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import vast_data  # noqa: E402

MARKET_TYPES = ["on-demand", "bid"]

# (column name in the table, field name in Vast's response)
COPIED_FIELDS = [
    ("offer_id", "id"),
    ("machine_id", "machine_id"),
    ("host_id", "host_id"),
    ("gpu_name", "gpu_name"),
    ("num_gpus", "num_gpus"),
    ("gpu_frac", "gpu_frac"),
    ("dph_base", "dph_base"),
    ("dph_total", "dph_total"),
    ("min_bid", "min_bid"),
    ("rentable", "rentable"),
    ("reliability", "reliability"),
    ("verification", "verification"),
    ("hosting_type", "hosting_type"),
    ("geolocation", "geolocation"),
    ("cluster_id", "cluster_id"),
    ("inet_down", "inet_down"),
    ("pci_gen", "pci_gen"),
    ("bw_nvlink", "bw_nvlink"),
    ("credit_discount_max", "credit_discount_max"),
    ("end_date", "end_date"),
]


def snapshot_time(run_id):
    """Run ids are UTC timestamps like 20261119T020347Z."""
    moment = datetime.strptime(run_id, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    return moment.isoformat()


def offer_row(run_id, market_type, offer):
    row = {
        "run_id": run_id,
        "snapshot_utc": snapshot_time(run_id),
        "market_type": market_type,
    }
    for column, field in COPIED_FIELDS:
        row[column] = offer.get(field)
    # GPU-only price per GPU-hour, comparable across offers of any size
    row["price_per_gpu"] = offer["dph_base"] / offer["num_gpus"] if offer.get("num_gpus") else None
    # gpu_ids identifies the physical GPUs, so the same card can be followed over time
    gpu_ids = []
    for gpu_id in offer.get("gpu_ids") or []:
        gpu_ids.append(str(gpu_id))
    row["gpu_ids"] = " ".join(gpu_ids)
    return row


def build(root):
    rows = []
    for market_type in MARKET_TYPES:
        for run_id in vast_data.complete_runs(root, market_type):
            for offer in vast_data.load_offers(root, market_type, run_id):
                rows.append(offer_row(run_id, market_type, offer))
    return rows


def main():
    parser = argparse.ArgumentParser(description="Flatten Vast snapshots into one table")
    parser.add_argument("--root", type=Path, default=Path("/root/data"))
    parser.add_argument("--out", type=Path, default=Path("/root/data/derived/vast_panel.csv"))
    args = parser.parse_args()

    rows = build(args.root)
    if not rows:
        sys.exit("No complete snapshots found.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    snapshots = set()
    for row in rows:
        snapshots.add((row["market_type"], row["run_id"]))
    print(f"{len(rows)} rows from {len(snapshots)} snapshots -> {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""One-off probe of the Vast.ai offer search. Not the collector.

Answers four questions before we design the real collector:
  1. Does search work without an API key?
  2. Can we see rented machines, or only rentable ones?
  3. Is there a result cap?
  4. Which fields come back, and is there a stable machine id?

Usage (on the droplet):
  pip install vastai
  python3 probe_vast.py              # writes probe_out/ and prints a summary
Then send me probe_out/summary.txt plus one sample record.

The query flags below are unverified guesses at Vast's query language.
A variant that errors is still useful information, so each one runs independently.
"""

import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

OUT = Path("probe_out")

# name -> extra CLI arguments for `vastai search offers`
VARIANTS = {
    "h100_default":       ["gpu_name=H100_SXM"],
    "h100_rentable_true": ["gpu_name=H100_SXM rentable=True"],
    "h100_rentable_false": ["gpu_name=H100_SXM rentable=False"],
    "h100_rentable_any":  ["gpu_name=H100_SXM rentable=any"],
    "h100_no_default":    ["gpu_name=H100_SXM", "--no-default"],
    "all_default":        [""],
    "all_no_default":     ["", "--no-default"],
    "all_big_limit":      ["", "--no-default", "--limit", "100000"],
}


def run_variant(name, extra_args):
    """Run one search, save raw stdout and stderr, return parsed offers or None."""
    command = ["vastai", "search", "offers"] + extra_args + ["--raw"]
    started = datetime.now(timezone.utc)
    completed = subprocess.run(command, capture_output=True, text=True, timeout=120)
    seconds = (datetime.now(timezone.utc) - started).total_seconds()

    # Keep the untouched output no matter what happened
    (OUT / f"{name}.stdout.json").write_text(completed.stdout)
    (OUT / f"{name}.stderr.txt").write_text(completed.stderr)

    header = f"== {name}  exit={completed.returncode}  {seconds:.1f}s  cmd={' '.join(command)}"
    if completed.returncode != 0:
        return header + f"\n   ERROR: {completed.stderr.strip()[:300]}", None

    try:
        offers = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return header + f"\n   NOT JSON: {completed.stdout[:300]}", None

    if isinstance(offers, dict):
        # Some versions may wrap results, e.g. {"offers": [...]}
        offers = offers.get("offers", offers)
    return header, offers


def describe(offers):
    """Summarize what an offer list tells us about the four questions."""
    lines = []
    if not isinstance(offers, list):
        lines.append(f"   unexpected type: {type(offers).__name__}")
        return lines

    lines.append(f"   offers returned: {len(offers)}")
    if not offers:
        return lines

    # Union of keys across all records, to see the full schema
    keys = set()
    for offer in offers:
        keys.update(offer.keys())
    lines.append(f"   distinct fields: {len(keys)}")

    # Candidate stable identifiers, and how many distinct values each has
    for id_field in ("id", "machine_id", "host_id", "ask_contract_id", "bundle_id"):
        if id_field in keys:
            distinct = len({offer.get(id_field) for offer in offers})
            lines.append(f"   {id_field}: {distinct} distinct")

    # Anything that looks like rental state
    for state_field in ("rentable", "rented", "is_rented", "num_gpus_rented", "cur_state", "next_state"):
        if state_field in keys:
            counts = Counter(str(offer.get(state_field)) for offer in offers)
            lines.append(f"   {state_field}: {dict(counts.most_common(6))}")

    # GPU mix, to judge coverage of the all-GPU queries
    if "gpu_name" in keys:
        top = Counter(offer.get("gpu_name") for offer in offers).most_common(8)
        lines.append(f"   top gpu_name: {top}")
    return lines


def main():
    OUT.mkdir(exist_ok=True)
    summary = [f"probe run at {datetime.now(timezone.utc).isoformat()}"]

    all_keys = set()
    sample = None
    for name, extra_args in VARIANTS.items():
        header, offers = run_variant(name, extra_args)
        summary.append(header)
        if offers is not None:
            summary.extend(describe(offers))
            if isinstance(offers, list) and offers:
                for offer in offers:
                    all_keys.update(offer.keys())
                if sample is None:
                    sample = offers[0]

    summary.append("== union of fields across all variants")
    summary.append("   " + ", ".join(sorted(all_keys)))

    (OUT / "summary.txt").write_text("\n".join(summary) + "\n")
    if sample is not None:
        (OUT / "sample_record.json").write_text(json.dumps(sample, indent=2, sort_keys=True))
    print("\n".join(summary))
    print(f"\nwrote {OUT}/summary.txt and {OUT}/sample_record.json")


if __name__ == "__main__":
    if subprocess.run(["which", "vastai"], capture_output=True).returncode != 0:
        sys.exit("vastai CLI not found. Run: pip install vastai")
    main()

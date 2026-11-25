#!/usr/bin/env python3
"""Do asking prices move on cards that nobody can rent, and does it matter?

Plain-language version
  A seller on Vast can change the asking price of a machine at any time, even
  while every card in it is taken. A renter who is already on the machine keeps
  the price from the moment they rented, so the new asking price is paid by
  nobody until the next rental. If an average is built from asking prices, such
  a change still moves it. This script measures how much.

Method
  1. Read every full snapshot (the ones that include the machines Vast's search
     leaves out). Count each physical card once.
  2. Follow each card across snapshots. Look only at the readings where the card
     was not rentable. If its highest asking price in those readings is more than
     20% above its lowest, flag the card.
  3. For each GPU model, compute the plain average price three ways:
       all cards            every listed card
       without those cards  flagged cards removed from every snapshot
       rentable cards only  only cards someone could rent at that moment
  4. For each way, report how far the average swings across snapshots, and the
     chance band for 20 random rentals (see noise_study.py).
  5. Control. Step 2 picks cards because their price moved, so removing them
     must reduce movement. As a comparison, also flag cards whose price moved
     more than 20% across readings where they WERE rentable (a price someone
     could act on) and remove those instead.
  6. Split every step between two snapshots into three parts that add up exactly:
       same cards, price moved, not rentable at both readings
       same cards, price moved, rentable at one reading or both
       cards entering or leaving the list (the mix effect)
     This part uses no threshold and no flagging, so it is not circular.
     Caution: Vast's lookups sometimes return only part of a machine, so some
     "entering or leaving" is the recorder missing cards, not cards moving.

What this does and does not show
  It shows how much an average of listed prices depends on which prices are
  counted. It says nothing about any index built from executed rentals.
  "Not rentable" is what the listing shows. That the card is rented, rather than
  held back by its seller, is an assumption.

Usage (on the droplet, reads only archived files, uses no Vast quota)
  python3 analysis/repricing.py --root /root/data
  python3 analysis/repricing.py --root /root/data --model "H100 SXM" --show-machines
"""

import argparse
import csv
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import noise_study  # noqa: E402
import vast_data  # noqa: E402

THRESHOLD = 0.20        # a card is flagged if its asking price moved more than this
RENTALS = 20            # rentals per hour used for the chance band
MIN_CARDS_MODEL = 10    # models with fewer cards than this are skipped
MIN_CARDS_BAND = 5      # a chance band needs at least this many cards
WAYS = ["all cards", "without those cards", "rentable cards only", "control"]


def card_key(card):
    return (card["gpu_name"], card["gpu_id"])


def load_cards_by_run(root, market_type):
    """Every full snapshot, as run_id -> list of cards (all GPU models)."""
    cards_by_run = {}
    for run_id in vast_data.complete_runs(root, market_type, full_only=True):
        offers = vast_data.load_offers(root, market_type, run_id)
        cards_by_run[run_id] = vast_data.physical_gpus(offers)
    return cards_by_run


def find_movers(cards_by_run, threshold=THRESHOLD, while_rentable=False):
    """Cards whose asking price moved more than `threshold` across one kind of reading.

    while_rentable=False uses only readings where the card was in no rentable
    offer. while_rentable=True uses only readings where it could be rented.
    A card with fewer than two such readings cannot be flagged.
    Returns card key -> {"low", "high", "changes", "readings", "machine_id", "host_id"}.
    """
    history = {}
    for run_id in sorted(cards_by_run):
        for card in cards_by_run[run_id]:
            if card["free"] != while_rentable:
                continue
            history.setdefault(card_key(card), []).append(card)

    flagged = {}
    for key, readings in history.items():
        if len(readings) < 2:
            continue
        prices = []
        for card in readings:
            prices.append(card["price"])
        low = min(prices)
        high = max(prices)
        if low <= 0 or high / low - 1 <= threshold:
            continue
        changes = 0
        for index in range(1, len(prices)):
            if abs(prices[index] - prices[index - 1]) > 0.005:
                changes += 1
        flagged[key] = {
            "low": low,
            "high": high,
            "changes": changes,
            "readings": len(readings),
            "machine_id": readings[-1]["machine_id"],
            "host_id": readings[-1]["host_id"],
        }
    return flagged


def find_repricers(cards_by_run, threshold=THRESHOLD):
    """Cards whose asking price moved more than `threshold` while not rentable."""
    return find_movers(cards_by_run, threshold, while_rentable=False)


def three_ways(cards, flagged, control=None):
    """Split one snapshot's cards into the sets being compared.

    control is the comparison group: cards whose price moved while rentable.
    """
    control = control or {}
    without = []
    rentable = []
    without_control = []
    for card in cards:
        if card_key(card) not in flagged:
            without.append(card)
        if card["free"]:
            rentable.append(card)
        if card_key(card) not in control:
            without_control.append(card)
    return {"all cards": cards, "without those cards": without, "rentable cards only": rentable,
            "control": without_control}


def average_price(cards):
    total = 0.0
    for card in cards:
        total += card["price"]
    return total / len(cards)


def swing(values):
    """How far the highest value sits above the lowest, as a share."""
    return max(values) / min(values) - 1


def snapshot_rows(model, cards_by_run, flagged, repeats, seed, control=None):
    """One row per snapshot: card counts, averages and chance bands, each way."""
    rows = []
    for run_id in sorted(cards_by_run):
        cards = []
        for card in cards_by_run[run_id]:
            if card["gpu_name"] == model:
                cards.append(card)
        if not cards:
            continue
        row = {"model": model, "run_id": run_id}
        for way, subset in three_ways(cards, flagged, control).items():
            row[f"cards: {way}"] = len(subset)
            row[f"average: {way}"] = average_price(subset) if subset else None
            if len(subset) >= MIN_CARDS_BAND:
                band = noise_study.simulate(noise_study.units(subset, "gpu"), RENTALS, 0.0,
                                            repeats, random.Random(seed))
                row[f"band low: {way}"] = band["p05"]
                row[f"band high: {way}"] = band["p95"]
            else:
                row[f"band low: {way}"] = None
                row[f"band high: {way}"] = None
        rows.append(row)
    return rows


def column(rows, name):
    values = []
    for row in rows:
        if row[name] is not None:
            values.append(row[name])
    return values


def summarize(model, rows, flagged, control=None):
    """Roll one model's snapshot rows up into a few numbers per way."""
    control_cards = 0
    for key in control or {}:
        if key[0] == model:
            control_cards += 1
    model_flagged = {}
    for key, info in flagged.items():
        if key[0] == model:
            model_flagged[key] = info
    machines = set()
    sellers = {}
    for info in model_flagged.values():
        machines.add(info["machine_id"])
        sellers[info["host_id"]] = sellers.get(info["host_id"], 0) + 1

    typical_cards = statistics.median(column(rows, "cards: all cards"))
    typical_flagged = typical_cards - statistics.median(column(rows, "cards: without those cards"))
    summary = {
        "model": model,
        "snapshots": len(rows),
        "cards": typical_cards,
        "flagged cards ever": len(model_flagged),
        "flagged share of supply": typical_flagged / typical_cards,
        "flagged machines": len(machines),
        "flagged sellers": len(sellers),
        "largest seller share of flagged": max(sellers.values()) / len(model_flagged) if model_flagged else 0.0,
        "control cards ever": control_cards,
    }
    for way in WAYS:
        averages = column(rows, f"average: {way}")
        lows = column(rows, f"band low: {way}")
        highs = column(rows, f"band high: {way}")
        summary[f"cards: {way}"] = statistics.median(column(rows, f"cards: {way}"))
        summary[f"average low: {way}"] = min(averages) if averages else None
        summary[f"average high: {way}"] = max(averages) if averages else None
        summary[f"average swing: {way}"] = swing(averages) if len(averages) >= 2 else None
        summary[f"band low median: {way}"] = statistics.median(lows) if lows else None
        summary[f"band high median: {way}"] = statistics.median(highs) if highs else None
        summary[f"band low widest: {way}"] = min(lows) if lows else None
        summary[f"band high widest: {way}"] = max(highs) if highs else None
        summary[f"snapshots with a band: {way}"] = len(lows)
    return summary


def percent(value, signed=False):
    if value is None:
        return "   n/a"
    if signed:
        return f"{100 * value:+5.1f}%"
    return f"{100 * value:5.1f}%"


def print_summary(summary, threshold):
    print(f"\n{summary['model']}   {summary['snapshots']} snapshots, about {summary['cards']:.0f} cards")
    if summary["flagged cards ever"] == 0:
        print(f"  no card's asking price moved more than {100 * threshold:.0f}% while not rentable")
    else:
        print(f"  cards whose asking price moved more than {100 * threshold:.0f}% while not rentable: "
              f"{summary['flagged cards ever']} ({100 * summary['flagged share of supply']:.0f}% of supply at a typical moment), "
              f"on {summary['flagged machines']} machines from {summary['flagged sellers']} sellers "
              f"(largest seller has {100 * summary['largest seller share of flagged']:.0f}% of them)")
    print("                         cards   average, lowest to highest   swing   20-rental band, typical   widest seen")
    for way in WAYS:
        low = summary[f"average low: {way}"]
        high = summary[f"average high: {way}"]
        if low is None:
            print(f"  {way:21}  {summary[f'cards: {way}']:5.0f}   no cards")
            continue
        print(f"  {way:21}  {summary[f'cards: {way}']:5.0f}   ${low:5.2f} to ${high:5.2f}             "
              f"{percent(summary[f'average swing: {way}'])}   "
              f"{percent(summary[f'band low median: {way}'], True)} {percent(summary[f'band high median: {way}'], True)}"
              f"          {percent(summary[f'band low widest: {way}'], True)} {percent(summary[f'band high widest: {way}'], True)}")
    print(f"  (control = without the {summary['control cards ever']} cards whose asking price moved more than "
          f"{100 * threshold:.0f}% while rentable)")
    thin = summary["snapshots with a band: rentable cards only"]
    if thin < summary["snapshots"]:
        print(f"  (rentable cards only: {summary['snapshots'] - thin} of {summary['snapshots']} snapshots had fewer than "
              f"{MIN_CARDS_BAND} rentable cards, so no band for those)")


def seller_overlap(flagged):
    """How concentrated the flagged cards are across sellers and GPU models. No ids returned."""
    cards_by_seller = {}
    models_by_seller = {}
    for key, info in flagged.items():
        seller = info["host_id"]
        cards_by_seller[seller] = cards_by_seller.get(seller, 0) + 1
        models_by_seller.setdefault(seller, set()).add(key[0])
    in_several = 0
    for models in models_by_seller.values():
        if len(models) >= 2:
            in_several += 1
    shares = sorted(cards_by_seller.values(), reverse=True)
    total = len(flagged)
    return {
        "flagged cards": total,
        "sellers": len(cards_by_seller),
        "sellers in two or more models": in_several,
        "largest seller share": shares[0] / total if total else 0.0,
        "top three sellers share": sum(shares[:3]) / total if total else 0.0,
    }


def decompose_steps(model, cards_by_run):
    """Split each step between consecutive snapshots into three parts that add up exactly.

    change in the average
      = price moves on cards present in both snapshots and not rentable in both
      + price moves on cards present in both and rentable in at least one
      + everything else, which is cards entering or leaving the list (mix)
    """
    steps = []
    previous_id = None
    previous = None
    for run_id in sorted(cards_by_run):
        current = {}
        for card in cards_by_run[run_id]:
            if card["gpu_name"] == model:
                current[card["gpu_id"]] = card
        if not current:
            continue
        if previous is not None:
            staying = []
            for gpu_id in current:
                if gpu_id in previous:
                    staying.append(gpu_id)
            average_before = average_price(list(previous.values()))
            average_now = average_price(list(current.values()))
            not_rentable_part = 0.0
            rentable_part = 0.0
            for gpu_id in staying:
                move = current[gpu_id]["price"] - previous[gpu_id]["price"]
                if not current[gpu_id]["free"] and not previous[gpu_id]["free"]:
                    not_rentable_part += move
                else:
                    rentable_part += move
            if staying:
                not_rentable_part /= len(staying)
                rentable_part /= len(staying)
            change = average_now - average_before
            steps.append({
                "model": model,
                "from_run": previous_id,
                "to_run": run_id,
                "average before": average_before,
                "average now": average_now,
                "change": change,
                "price moved, not rentable": not_rentable_part,
                "price moved, rentable": rentable_part,
                "entering or leaving": change - not_rentable_part - rentable_part,
                "cards staying": len(staying),
                "cards entered": len(current) - len(staying),
                "cards left": len(previous) - len(staying),
            })
        previous_id = run_id
        previous = current
    return steps


STEP_PARTS = ["change", "price moved, not rentable", "price moved, rentable", "entering or leaving"]


def summarize_steps(steps):
    """Typical size of each part of a step, as a share of the average price level."""
    level = statistics.fmean(step["average before"] for step in steps)
    summary = {"model": steps[0]["model"], "steps": len(steps)}
    for part in STEP_PARTS:
        summary[part] = statistics.fmean(abs(step[part]) for step in steps) / level
    summary["cards entered or left per step"] = statistics.fmean(
        step["cards entered"] + step["cards left"] for step in steps)
    return summary


def print_steps(step_summary):
    print(f"  what moved the average between one snapshot and the next (typical size, as a share of the average price):")
    labels = {
        "change": "whole step",
        "price moved, not rentable": "same cards, price moved, not rentable",
        "price moved, rentable": "same cards, price moved, rentable",
        "entering or leaving": "cards entering or leaving the list",
    }
    for part in STEP_PARTS:
        print(f"    {labels[part]:40} {100 * step_summary[part]:5.2f}%")
    print(f"    cards entering or leaving per step: about {step_summary['cards entered or left per step']:.0f}")
    print("    (the three parts add up to the whole step exactly, sign included. Their sizes can add up to more"
          " when they pull in opposite directions.)")


def print_machines(model, flagged):
    """For private inspection only. Machine ids are not for publication."""
    by_machine = {}
    for key, info in flagged.items():
        if key[0] != model:
            continue
        entry = by_machine.setdefault(info["machine_id"], {"cards": 0, "low": info["low"], "high": info["high"],
                                                           "changes": 0})
        entry["cards"] += 1
        entry["low"] = min(entry["low"], info["low"])
        entry["high"] = max(entry["high"], info["high"])
        entry["changes"] = max(entry["changes"], info["changes"])
    for machine_id in sorted(by_machine, key=lambda m: -by_machine[m]["cards"]):
        entry = by_machine[machine_id]
        print(f"    machine {machine_id:>8}  {entry['cards']:2d} cards  ${entry['low']:.2f} to ${entry['high']:.2f}  "
              f"price changed {entry['changes']} times")


def write_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Asking prices that move while nobody can rent")
    parser.add_argument("--root", type=Path, default=Path("/root/data"))
    parser.add_argument("--market", default="on-demand")
    parser.add_argument("--model", default=None, help="one GPU model, default every model found")
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--show-machines", action="store_true",
                        help="list the flagged machines (private inspection, do not publish ids)")
    parser.add_argument("--out", type=Path, default=Path("/root/data/analysis"))
    args = parser.parse_args()

    cards_by_run = load_cards_by_run(args.root, args.market)
    if len(cards_by_run) < 2:
        sys.exit("Need at least two full snapshots.")
    run_ids = sorted(cards_by_run)
    print(f"{len(run_ids)} full snapshots, {run_ids[0]} to {run_ids[-1]}, {args.market}")

    flagged = find_repricers(cards_by_run, args.threshold)
    control = find_movers(cards_by_run, args.threshold, while_rentable=True)
    models = set()
    for cards in cards_by_run.values():
        for card in cards:
            models.add(card["gpu_name"])
    if args.model:
        models = {args.model}

    all_rows = []
    all_steps = []
    summaries = []
    for model in sorted(models):
        rows = snapshot_rows(model, cards_by_run, flagged, args.repeats, args.seed, control)
        if not rows or statistics.median(column(rows, "cards: all cards")) < MIN_CARDS_MODEL:
            print(f"\n{model}   fewer than {MIN_CARDS_MODEL} cards, skipped")
            continue
        summary = summarize(model, rows, flagged, control)
        print_summary(summary, args.threshold)
        steps = decompose_steps(model, cards_by_run)
        if steps:
            print_steps(summarize_steps(steps))
            all_steps.extend(steps)
        if args.show_machines:
            print_machines(model, flagged)
        all_rows.extend(rows)
        summaries.append(summary)

    if not summaries:
        sys.exit("No model had enough cards.")
    overlap = seller_overlap(flagged)
    if overlap["flagged cards"]:
        print(f"\nacross all models: {overlap['flagged cards']} flagged cards from {overlap['sellers']} sellers. "
              f"{overlap['sellers in two or more models']} of those sellers show up in two or more GPU models. "
              f"The largest holds {100 * overlap['largest seller share']:.0f}% of all flagged cards, "
              f"the top three hold {100 * overlap['top three sellers share']:.0f}%.")

    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(all_rows, args.out / "repricing_by_snapshot.csv")
    write_csv(summaries, args.out / "repricing_by_model.csv")
    if all_steps:
        write_csv(all_steps, args.out / "repricing_steps.csv")
    print(f"\ntables in {args.out}: repricing_by_snapshot.csv, repricing_by_model.csv, repricing_steps.csv")


if __name__ == "__main__":
    main()

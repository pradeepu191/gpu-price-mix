#!/usr/bin/env python3
"""How much does an hourly GPU price index move purely by chance?

Plain-language version
  An hourly price index averages the rentals that happened in that hour. When
  only a few rentals happen, which ones they were is partly luck, so the average
  can jump even though no price changed. This script measures how big those
  luck-driven jumps are, depending on how many rentals sit behind the hour.

Method
  1. Take one snapshot of every listed card of one GPU model on Vast. Treat it
     as the whole market at a single moment, so by construction no price changes.
     Each physical card is counted once, even though Vast lists most machines
     several times as overlapping bundles.
  2. Pretend N rentals happened in an hour: pick N at random (repeats allowed).
  3. Compute the index on those N rentals the way Ornn describes OCPI: an average
     weighted by GPU count, with the most extreme prices pulled in to a cutoff
     (a "winsorized" mean). Ornn does not publish its cutoff, so several are tried.
  4. Compare with the same index computed on the whole snapshot.
  5. Repeat thousands of times. The spread of the differences is pure chance.

What a rental is
  gpu level:     each rental is a single card. Optimistic: many small, independent picks.
  machine level: each rental is a whole machine, weighted by its cards. Cautious:
                 fewer, bigger picks. Real rentals sit between the two.

What this does and does not show
  It measures a property of the averaging method on a realistic spread of prices.
  It is NOT an estimate of OCPI's own error: Vast's listed prices are not the
  rentals Ornn sees, and Ornn's rental counts per hour are not public.

Usage (on the droplet, reads only archived files, uses no Vast quota)
  One snapshot, full tables and chart:
    python3 analysis/noise_study.py --root /root/data --model "H100 SXM"
  Every snapshot so far, a few headline numbers each, to see if they hold up:
    python3 analysis/noise_study.py --root /root/data --model "H100 SXM" --all-runs
"""

import argparse
import csv
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import vast_data  # noqa: E402

RENTAL_COUNTS = [1, 2, 3, 5, 8, 13, 20, 30, 50, 80, 130, 200]
CUTOFFS = [0.0, 0.05, 0.10]     # share of GPU weight pulled in from each tail
CHART_CUTOFF = 0.0              # the chart shows the plain average

# Two ways a "rental" can be modeled. The truth sits between them.
LEVELS = {
    "gpu": "each rental is one GPU card, picked at random",
    "machine": "each rental is a whole machine, picked at random, weighted by its GPU count",
}


def weighted_quantile(rentals, share, method="step"):
    """Price below which `share` of the total GPU weight sits.

    rentals is a list of (price, gpu_count).
    step:   walk up the sorted prices adding GPU counts, and stop at the first
            price where the running total reaches the share. Jumps from one
            offer's price to the next.
    smooth: place each offer at the middle of its slice of GPU weight and draw
            straight lines between offers, so the answer moves gradually.
    """
    ordered = sorted(rentals)
    total_weight = 0
    for price, weight in ordered:
        total_weight += weight

    if method == "step":
        target = share * total_weight
        running = 0
        for price, weight in ordered:
            running += weight
            if running >= target:
                return price
        return ordered[-1][0]

    # smooth: each offer sits at the midpoint of its weight slice, from 0 to 1
    positions = []
    running = 0
    for price, weight in ordered:
        positions.append(((running + weight / 2) / total_weight, price))
        running += weight
    if share <= positions[0][0]:
        return positions[0][1]
    if share >= positions[-1][0]:
        return positions[-1][1]
    for index in range(1, len(positions)):
        left_position, left_price = positions[index - 1]
        right_position, right_price = positions[index]
        if share <= right_position:
            if right_position == left_position:
                return right_price
            fraction = (share - left_position) / (right_position - left_position)
            return left_price + fraction * (right_price - left_price)
    return positions[-1][1]


def winsorized_weighted_mean(rentals, cutoff, method="step"):
    """Average price weighted by GPU count, after pulling extreme prices in.

    Prices below the `cutoff` quantile are raised to it, and prices above the
    (1 - cutoff) quantile are lowered to it. Nothing is thrown away. With
    cutoff 0 this is the plain GPU-weighted average.
    """
    if cutoff > 0:
        low = weighted_quantile(rentals, cutoff, method)
        high = weighted_quantile(rentals, 1 - cutoff, method)
    else:
        low = None
        high = None

    weighted_sum = 0.0
    total_weight = 0
    for price, weight in rentals:
        clamped = price
        if low is not None and clamped < low:
            clamped = low
        if high is not None and clamped > high:
            clamped = high
        weighted_sum += clamped * weight
        total_weight += weight
    return weighted_sum / total_weight


def percentile(sorted_values, share):
    """Simple percentile of an already sorted list (nearest rank)."""
    index = round(share * (len(sorted_values) - 1))
    return sorted_values[index]


def simulate(rentals, rental_count, cutoff, repeats, rng, method="step"):
    """Relative error of the index on N random rentals, versus the whole snapshot.

    bias is the average error: a thin hour that leans high or low on average.
    p05 to p95 is the range holding 90% of chance outcomes.
    """
    true_value = winsorized_weighted_mean(rentals, cutoff, method)
    errors = []
    for _ in range(repeats):
        sample = []
        for _ in range(rental_count):
            sample.append(rng.choice(rentals))
        estimate = winsorized_weighted_mean(sample, cutoff, method)
        errors.append(estimate / true_value - 1)
    errors.sort()
    absolute_errors = []
    for error in errors:
        absolute_errors.append(abs(error))
    return {
        "true_value": true_value,
        "bias": statistics.fmean(errors),
        "p05": percentile(errors, 0.05),
        "p50": percentile(errors, 0.50),
        "p95": percentile(errors, 0.95),
        "typical_abs": statistics.median(absolute_errors),
    }


def describe_tails(rentals, cutoff, method):
    """Show where the cutoffs land and which offers they pull in, to see if the tails are clumpy."""
    low = weighted_quantile(rentals, cutoff, method)
    high = weighted_quantile(rentals, 1 - cutoff, method)
    total_weight = 0
    for price, weight in rentals:
        total_weight += weight
    above = []
    below = []
    for price, weight in sorted(rentals):
        if price >= high:
            above.append((round(price, 2), weight))
        if price <= low:
            below.append((round(price, 2), weight))
    above_weight = 0
    for price, weight in above:
        above_weight += weight
    below_weight = 0
    for price, weight in below:
        below_weight += weight
    print(f"  {method:6} cutoff {cutoff:.2f}: pull-in range ${low:.2f} to ${high:.2f}")
    print(f"     at or below ${low:.2f}: {len(below)} units, {100 * below_weight / total_weight:.1f}% of GPUs {below[:6]}")
    print(f"     at or above ${high:.2f}: {len(above)} units, {100 * above_weight / total_weight:.1f}% of GPUs {above[-8:]}")


def signed_percent(value, _position):
    """Axis label like +20% or -20%, and 0% at the middle."""
    if value == 0:
        return "0%"
    return f"{value:+.0f}%"


def draw_chart(rows, model, population_label, out_path):
    """One chart: the 90% range of chance error against rentals per hour."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker
    except ImportError:
        print("matplotlib not installed, skipping the chart (pip install matplotlib)")
        return

    bands = {}
    for level in LEVELS:
        counts = []
        lows = []
        highs = []
        for row in rows:
            if row["cutoff"] == CHART_CUTOFF and row["level"] == level:
                counts.append(row["rentals_per_hour"])
                lows.append(100 * row["p05"])
                highs.append(100 * row["p95"])
        bands[level] = (counts, lows, highs)

    surface = "#fcfcfb"
    ink = "#0b0b0b"
    ink_secondary = "#52514e"
    colors = {"gpu": "#2a78d6", "machine": "#eb6834"}
    names = {"gpu": "Each rental is one card", "machine": "Each rental is a whole machine"}

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    for level in ("machine", "gpu"):
        counts, lows, highs = bands[level]
        if level == "gpu":
            ax.fill_between(counts, lows, highs, color=colors[level], alpha=0.14, linewidth=0)
        ax.plot(counts, highs, color=colors[level], linewidth=2, label=names[level])
        ax.plot(counts, lows, color=colors[level], linewidth=2)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=ink_secondary, loc="upper right")
    ax.axhline(0, color=ink_secondary, linewidth=0.8)

    ax.set_xscale("log")
    ax.set_xticks(counts)
    ax.set_xticklabels([str(count) for count in counts])
    ax.set_xlabel("Rentals behind one hourly price", color=ink_secondary)
    ax.set_ylabel("Error from chance alone", color=ink_secondary)
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(signed_percent))
    fig.text(0.012, 0.965, f"How far an hourly {model} price can move with no real price change",
             color=ink, fontsize=11.5, weight="bold", va="top")
    if CHART_CUTOFF > 0:
        cutoff_note = f"The most extreme {int(CHART_CUTOFF * 100)}% of prices on each side are pulled in."
    else:
        cutoff_note = "Plain average, each card counted once."
    fig.text(0.012, 0.905, f"90% of chance outcomes fall inside the band. {population_label}. {cutoff_note}",
             color=ink_secondary, fontsize=8.5, va="top")

    ax.grid(axis="y", color="#e6e5e0", linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=ink_secondary)

    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(out_path, facecolor=surface)
    print(f"chart: {out_path}")


def units(cards, level):
    """Turn cards into (price, weight) pairs for one way of modeling a rental."""
    result = []
    if level == "gpu":
        for card in cards:
            result.append((card["price"], 1))
    else:
        for machine in vast_data.machines(cards):
            result.append((machine["price"], machine["gpus"]))
    return result


def select_cards(offers, model, occupied_only):
    cards = vast_data.physical_gpus(offers, model)
    if occupied_only:
        kept = []
        for card in cards:
            if card["occupied"]:
                kept.append(card)
        cards = kept
    return cards


def price_shape(prices_and_weights):
    """Simple description of the price spread, weighted."""
    lowest = min(price for price, weight in prices_and_weights)
    total = 0
    near_floor = 0
    for price, weight in prices_and_weights:
        total += weight
        if price <= lowest * 1.01:
            near_floor += weight
    return {
        "p10": weighted_quantile(prices_and_weights, 0.10),
        "p50": weighted_quantile(prices_and_weights, 0.50),
        "p90": weighted_quantile(prices_and_weights, 0.90),
        "floor_share": near_floor / total,
    }


def host_shares(cards):
    """Each host's share of all cards, largest first."""
    counts = {}
    for card in cards:
        counts[card["host_id"]] = counts.get(card["host_id"], 0) + 1
    shares = []
    for host, count in counts.items():
        shares.append((count / len(cards), host))
    shares.sort(reverse=True)
    return shares


def describe_market(cards):
    machine_list = vast_data.machines(cards)
    occupied = 0
    for card in cards:
        if card["occupied"]:
            occupied += 1
    shape = price_shape(units(cards, "gpu"))
    shares = host_shares(cards)
    print(f"cards: {len(cards)}   machines: {len(machine_list)}   hosts: {len(shares)}   "
          f"occupied: {100 * occupied / len(cards):.0f}%")
    print(f"price per GPU-hour: 10th pct ${shape['p10']:.2f}, median ${shape['p50']:.2f}, "
          f"90th pct ${shape['p90']:.2f}, lowest ${min(card['price'] for card in cards):.2f}")
    print(f"cards within 1% of the lowest price: {100 * shape['floor_share']:.0f}%")
    top = shares[:5]
    print("largest hosts' share of cards: " + ", ".join(f"{100 * share:.0f}%" for share, host in top)
          + f"   (top two together: {100 * (shares[0][0] + (shares[1][0] if len(shares) > 1 else 0)):.0f}%)")


def print_rows(rows):
    print("\nlevel    cutoff  rentals    bias   5th pct   median  95th pct   typical |error|")
    for row in rows:
        print(f"{row['level']:8} {row['cutoff']:6.2f}  {row['rentals_per_hour']:7d}  {100 * row['bias']:+5.1f}%  "
              f"{100 * row['p05']:+7.1f}%  {100 * row['p50']:+6.1f}%  {100 * row['p95']:+7.1f}%  "
              f"{100 * row['typical_abs']:10.1f}%")


def write_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def study(cards, repeats, seed):
    """All simulation rows for one snapshot."""
    rows = []
    for level in LEVELS:
        pairs = units(cards, level)
        for cutoff in CUTOFFS:
            for count in RENTAL_COUNTS:
                rng = random.Random(seed + count)
                result = simulate(pairs, count, cutoff, repeats, rng)
                row = {"level": level, "cutoff": cutoff, "rentals_per_hour": count}
                row.update(result)
                rows.append(row)
    return rows


def run_single(args, run_id):
    offers = vast_data.load_offers(args.root, args.market, run_id)
    cards = select_cards(offers, args.model, args.occupied_only)
    if len(cards) < 10:
        sys.exit(f"Only {len(cards)} usable {args.model} cards in run {run_id}; too few to study.")

    population_label = "Occupied cards only" if args.occupied_only else "All listed cards"
    print(f"run {run_id}, {args.market}, {args.model}, {population_label.lower()}")
    describe_market(cards)

    print("\nwhere the cutoffs land (machine level, where big machines make the tails lumpy):")
    machine_pairs = units(cards, "machine")
    for cutoff in CUTOFFS:
        if cutoff > 0:
            describe_tails(machine_pairs, cutoff, "step")

    rows = study(cards, args.repeats, args.seed)
    print_rows(rows)

    args.out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.model.replace(' ', '_')}_{run_id}{'_occupied' if args.occupied_only else ''}"
    csv_path = args.out / f"noise_{tag}.csv"
    write_csv(rows, csv_path)
    print(f"\ntable: {csv_path}")
    draw_chart(rows, args.model, population_label, args.out / f"noise_{tag}.png")


def headline(cards, repeats, seed):
    """A few numbers per snapshot, to check stability over time."""
    gpu_pairs = units(cards, "gpu")
    machine_pairs = units(cards, "machine")
    shape = price_shape(gpu_pairs)
    shares = host_shares(cards)
    occupied = 0
    for card in cards:
        if card["occupied"]:
            occupied += 1
    gpu_20 = simulate(gpu_pairs, 20, 0.0, repeats, random.Random(seed))
    machine_20 = simulate(machine_pairs, 20, 0.0, repeats, random.Random(seed))
    gpu_3 = simulate(gpu_pairs, 3, 0.0, repeats, random.Random(seed))
    machine_cut10 = simulate(machine_pairs, 20, 0.10, repeats, random.Random(seed))
    return {
        "cards": len(cards),
        "machines": len(machine_pairs),
        "hosts": len(shares),
        "top2_host_share": shares[0][0] + (shares[1][0] if len(shares) > 1 else 0),
        "occupied_share": occupied / len(cards),
        "price_p50": shape["p50"],
        "floor_share": shape["floor_share"],
        "gpu20_low": gpu_20["p05"],
        "gpu20_high": gpu_20["p95"],
        "machine20_low": machine_20["p05"],
        "machine20_high": machine_20["p95"],
        "gpu3_median_error": gpu_3["p50"],
        "machine20_cut10_bias": machine_cut10["bias"],
    }


def run_all(args):
    rows = []
    run_ids = vast_data.complete_runs(args.root, args.market, full_only=not args.include_partial)
    if not args.include_partial:
        skipped = len(vast_data.complete_runs(args.root, args.market)) - len(run_ids)
        print(f"using {len(run_ids)} full snapshots (hidden machines included). "
              f"{skipped} earlier partial snapshots left out. Add --include-partial to see them too.\n")
    for run_id in run_ids:
        offers = vast_data.load_offers(args.root, args.market, run_id)
        cards = select_cards(offers, args.model, args.occupied_only)
        if len(cards) < 10:
            print(f"{run_id}: only {len(cards)} cards, skipped")
            continue
        row = {"run_id": run_id}
        row.update(headline(cards, args.repeats, args.seed))
        rows.append(row)
    if not rows:
        sys.exit("No usable snapshots yet.")
    print_summary(rows, args.model, args.market)
    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / f"noise_by_snapshot_{args.model.replace(' ', '_')}.csv"
    write_csv(rows, csv_path)
    print(f"\ntable: {csv_path}")


def print_summary(rows, model, market):
    print(f"{len(rows)} snapshots of {model}, {market}\n")
    print("run_id            cards mach hosts top2  occ%  median  20 rentals: per card   per machine   "
          "median@3  lean@cut10")
    for row in rows:
        print(f"{row['run_id']}  {row['cards']:5d} {row['machines']:4d} {row['hosts']:5d} "
              f"{100 * row['top2_host_share']:4.0f}% {100 * row['occupied_share']:4.0f}%  ${row['price_p50']:5.2f}"
              f"     {100 * row['gpu20_low']:+5.1f}% {100 * row['gpu20_high']:+5.1f}%"
              f"   {100 * row['machine20_low']:+5.1f}% {100 * row['machine20_high']:+5.1f}%"
              f"   {100 * row['gpu3_median_error']:+5.1f}%    {100 * row['machine20_cut10_bias']:+5.1f}%")
    print("\nacross snapshots (min / median / max):")
    for key in ("gpu20_low", "gpu20_high", "machine20_low", "machine20_high", "gpu3_median_error",
                "machine20_cut10_bias", "top2_host_share", "occupied_share", "floor_share"):
        values = []
        for row in rows:
            values.append(row[key])
        values.sort()
        print(f"  {key:22} {100 * values[0]:+6.1f}%  {100 * statistics.median(values):+6.1f}%  {100 * values[-1]:+6.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Chance error of an hourly GPU price index")
    parser.add_argument("--root", type=Path, default=Path("/root/data"))
    parser.add_argument("--model", default="H100 SXM")
    parser.add_argument("--market", default="on-demand")
    parser.add_argument("--run-id", default=None, help="defaults to the newest complete run")
    parser.add_argument("--all-runs", action="store_true", help="summarize every complete snapshot")
    parser.add_argument("--repeats", type=int, default=None,
                        help="random repeats per setting (default 20000, or 4000 with --all-runs)")
    parser.add_argument("--include-partial", action="store_true",
                        help="also use snapshots from before the collector asked for hidden machines")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--occupied-only", action="store_true",
                        help="use only cards currently in no rentable offer, the closest thing to real rentals")
    parser.add_argument("--out", type=Path, default=Path("/root/data/analysis"))
    args = parser.parse_args()

    if args.all_runs:
        if args.repeats is None:
            args.repeats = 4000
        run_all(args)
        return
    if args.repeats is None:
        args.repeats = 20000
    run_id = args.run_id or vast_data.latest_complete_run(args.root, args.market, full_only=not args.include_partial)
    run_single(args, run_id)


if __name__ == "__main__":
    main()

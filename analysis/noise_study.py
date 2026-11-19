#!/usr/bin/env python3
"""How much does an hourly GPU price index move purely by chance?

Plain-language version
  An hourly price index averages the rentals that happened in that hour. When
  only a few rentals happen, which ones they were is partly luck, so the average
  can jump even though no price changed. This script measures how big those
  luck-driven jumps are, depending on how many rentals sit behind the hour.

Method
  1. Take one snapshot of every offer for one GPU model on Vast. Treat it as the
     whole market at a single moment, so by construction no price changes.
  2. Pretend N rentals happened in an hour: pick N offers at random (repeats
     allowed, every offer equally likely).
  3. Compute the index on those N rentals the way Ornn describes OCPI: an average
     weighted by GPU count, with the most extreme prices pulled in to a cutoff
     (a "winsorized" mean). Ornn does not publish its cutoff, so several are tried.
  4. Compare with the same index computed on the whole snapshot.
  5. Repeat thousands of times. The spread of the differences is pure chance.

What this does and does not show
  It measures a property of the averaging method on a realistic spread of prices.
  It is NOT an estimate of OCPI's own error: Vast's asking prices are not the
  rentals Ornn sees, and Ornn's rental counts per hour are not public.

Usage (on the droplet, reads only archived files, uses no Vast quota)
  python3 analysis/noise_study.py --root /root/data --model "H100 SXM" --out /root/data/analysis
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
QUANTILE_METHODS = ["step", "smooth"]
CHART_CUTOFF = 0.0              # the chart shows the plain GPU-weighted average


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
    print(f"     at or below ${low:.2f}: {len(below)} offers, {100 * below_weight / total_weight:.1f}% of GPUs {below[:6]}")
    print(f"     at or above ${high:.2f}: {len(above)} offers, {100 * above_weight / total_weight:.1f}% of GPUs {above[-8:]}")


def describe(name, rentals):
    prices = []
    gpu_total = 0
    for price, weight in rentals:
        prices.append(price)
        gpu_total += weight
    prices.sort()
    print(f"{name}: {len(rentals)} offers, {gpu_total} GPUs, "
          f"price $/GPU-hr min {prices[0]:.2f}, median {statistics.median(prices):.2f}, max {prices[-1]:.2f}")


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

    chart_rows = []
    for row in rows:
        if row["cutoff"] == CHART_CUTOFF and row["method"] == "step":
            chart_rows.append(row)
    counts = []
    lows = []
    highs = []
    for row in chart_rows:
        counts.append(row["rentals_per_hour"])
        lows.append(100 * row["p05"])
        highs.append(100 * row["p95"])

    surface = "#fcfcfb"
    ink = "#0b0b0b"
    ink_secondary = "#52514e"
    series = "#2a78d6"

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    ax.fill_between(counts, lows, highs, color=series, alpha=0.18, linewidth=0)
    ax.plot(counts, highs, color=series, linewidth=2)
    ax.plot(counts, lows, color=series, linewidth=2)
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
        cutoff_note = "Plain average weighted by GPU count."
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


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--root", type=Path, default=Path("/root/data"))
    parser.add_argument("--model", default="H100 SXM")
    parser.add_argument("--market", default="on-demand")
    parser.add_argument("--run-id", default=None, help="defaults to the newest complete run")
    parser.add_argument("--repeats", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--occupied-only", action="store_true",
                        help="use only machines currently rented, the closest thing to real rentals")
    parser.add_argument("--out", type=Path, default=Path("/root/data/analysis"))
    args = parser.parse_args()

    run_id = args.run_id or vast_data.latest_complete_run(args.root, args.market)
    offers = vast_data.load_offers(args.root, args.market, run_id)

    rentals = []
    occupied_count = 0
    for offer in offers:
        if offer["gpu_name"] != args.model:
            continue
        if offer["num_gpus"] <= 0 or offer["dph_base"] is None:
            continue
        occupied = vast_data.is_occupied(offer)
        if occupied:
            occupied_count += 1
        if args.occupied_only and not occupied:
            continue
        rentals.append((vast_data.price_per_gpu_hour(offer), offer["num_gpus"]))

    if len(rentals) < 10:
        sys.exit(f"Only {len(rentals)} usable {args.model} offers in run {run_id}; too few to study.")

    population_label = "Occupied machines only" if args.occupied_only else "All listed offers"
    print(f"run {run_id}, {args.market}, {args.model}, {population_label.lower()}")
    print(f"occupied (rentable=False): {occupied_count}")
    describe("population", rentals)

    print("\nwhere the cutoffs land:")
    for method in QUANTILE_METHODS:
        for cutoff in CUTOFFS:
            if cutoff > 0:
                describe_tails(rentals, cutoff, method)

    rows = []
    for method in QUANTILE_METHODS:
        for cutoff in CUTOFFS:
            if cutoff == 0 and method != "step":
                continue   # with no cutoff the method makes no difference
            for count in RENTAL_COUNTS:
                # same random draws for every setting, so differences come from the method
                rng = random.Random(args.seed + count)
                result = simulate(rentals, count, cutoff, args.repeats, rng, method)
                row = {"method": method, "cutoff": cutoff, "rentals_per_hour": count}
                row.update(result)
                rows.append(row)

    print("\nmethod cutoff  rentals    bias   5th pct   median  95th pct   typical |error|")
    for row in rows:
        print(f"{row['method']:6} {row['cutoff']:6.2f}  {row['rentals_per_hour']:7d}  {100 * row['bias']:+5.1f}%  "
              f"{100 * row['p05']:+7.1f}%  {100 * row['p50']:+6.1f}%  {100 * row['p95']:+7.1f}%  "
              f"{100 * row['typical_abs']:10.1f}%")

    args.out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.model.replace(' ', '_')}_{run_id}{'_occupied' if args.occupied_only else ''}"
    csv_path = args.out / f"noise_{tag}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\ntable: {csv_path}")
    draw_chart(rows, args.model, population_label, args.out / f"noise_{tag}.png")


if __name__ == "__main__":
    main()

# gpu-price-mix

Companies rent GPUs (the chips that run AI) by the hour, and some groups now publish a daily "average rental price" for each kind of GPU, much like an average house price for a city. Contracts can even pay out based on those averages.

But an average can move for reasons that have nothing to do with prices changing. If more expensive machines happen to get rented today, the average goes up even though nobody raised their price.

This project collects regular snapshots of every datacenter GPU listed on [Vast.ai](https://vast.ai), a public rental marketplace, and tries to separate three things:

1. **Real price changes:** the same machines got cheaper or more expensive.
2. **Changes in what got rented:** prices stayed the same, but the average shifted because different kinds of machines were rented. Economists call this a *mix* or *composition* effect.
3. **Luck:** when only a handful of rentals happen in an hour, which ones they were is partly chance, so the average jumps around even if nothing changed.

It is a work in progress. Everything below is early and may change as more data comes in.

## What is collected

| Source | What | How often |
| --- | --- | --- |
| Vast.ai | Every offer for 11 datacenter GPU models (A100 through B300), both free and currently rented machines, with price, GPU count, location, reliability and seller | Every 2 hours for on-demand rentals, every 6 hours for interruptible ones |
| [Ornn](https://data.ornn.com) | Ornn's public GPU price index (OCPI) and token price index (OTPI): current hourly values, daily settlement prices, and the recent history its free tier offers | Hourly, daily, and weekly |

A few terms:

- **On-demand:** rent any time, stop any time, pay by the hour, never interrupted.
- **Interruptible:** cheaper, but the machine can be taken back mid-job. Vast calls these "bid" offers.
- **Index:** a single published number meant to summarize a market's price, here dollars per GPU per hour.

**The raw data is not in this repository.** It stays on the collection server. Vast's and Ornn's terms limit redistribution, and I have asked Vast about research use. Only code and summary results are published here.

## Early findings

From 11 snapshots taken every 2 hours on 19 November 2026, on-demand rentals, H100 SXM unless stated otherwise. These are early and not yet checked over longer periods.

- **Vast lists the same machine several times at once.** One 8-GPU machine often appears as a 1-GPU, 2-GPU, 4-GPU and 8-GPU offer at the same time. Counting offers overstates the number of GPUs by about 45%, so this project counts each physical card once.
- **A few sellers dominate.** About 140 H100 SXM cards on 35 machines came from 16 sellers. The two largest held 36% to 45% of the cards in every snapshot, so one seller changing its price can move the average noticeably.
- **Most listed cards are busy.** 83% to 91% of cards were not available to rent, which most likely means someone was renting them. Cards switching between free and taken from one snapshot to the next support this, but do not prove it.
- **Vast's search does not return every live machine.** In a direct test, 10 of 10 machines missing from a search answered when asked for by ID. A snapshot shows roughly two thirds of the machines, and which ones it shows changes each time. Most of the 2-hour wobble in the average price came from this, not from prices or real changes in supply. The counts in this section come from these partial lists and will be redone once the collector asks for missing machines directly.
- **Hourly averages built from few rentals are noisy.** Simulating hours with no price changes at all, an average built from 20 rentals landed between about 11% below and 13% above the true average in 90% of hours, and up to 18% below and 25% above in the most spread-out snapshot. Around 200 rentals keep it within about 3%.
- **Pulling in extreme prices made little consistent difference.** Indices often pull the most extreme prices in toward a cutoff before averaging (called *winsorizing*). Here it neither helped nor hurt reliably across snapshots.
- **Price levels roughly match a published index for some chips.** Vast's average was within a few percent of Ornn's index for A100, B200 and B300, and about 15% below it for H100 SXM.

**What these findings are not:** they describe Vast's listed prices and a simulation of an averaging method. They are not a measurement of any published index's accuracy. Listed prices are not the same as completed rentals, Vast is not the whole market, and the number of rentals behind any published hourly price is not public.

## How to run it

Requires Python 3.10 or later. The collectors were built to run on a small always-on Linux server.

```bash
git clone https://github.com/pradeepu191/gpu-price-mix.git
cd gpu-price-mix
python3 -m venv ~/venv && source ~/venv/bin/activate
pip install vastai matplotlib
vastai set api-key <your Vast API key>   # a read-only key is enough

python3 -m unittest discover -s tests     # all tests should pass

# one snapshot of each source
python3 collectors/vast/collect_vast.py --root /root/data --force-bid
python3 collectors/ornn/collect_ornn.py backfill --root /root/data/ornn

# the noise study, on the newest snapshot or on every snapshot so far
python3 analysis/noise_study.py --root /root/data --model "H100 SXM"
python3 analysis/noise_study.py --root /root/data --model "H100 SXM" --all-runs
```

Vast limits how many search results each account can fetch per day, which is why the collector covers datacenter GPUs only and runs every 2 hours rather than every hour.

To run the collectors on a schedule, copy the files in `deploy/` to `/etc/systemd/system/` and enable the timers. The comments in each file explain the schedule.

## How the collectors keep the data trustworthy

- Every response is saved exactly as received, before any processing, so nothing is lost if a source changes its format.
- Every request is logged, including failures, so a gap in the data always has a recorded reason.
- Vast often lists one machine several times as overlapping bundles, so the analysis follows each physical GPU card by its ID and counts it once.
- Vast returns at most 512 results per search and does not reliably return the cheapest ones first, so the collector queries one GPU model at a time and splits any full result by price range until nothing is cut off.
- Ornn's daily prices are dated by New York midnight, which falls at 04:00 or 05:00 UTC depending on daylight saving. Comparisons use New York dates.

## Layout

```
collectors/vast/   Vast offer collector, and the probe used to explore the API
collectors/ornn/   Ornn index collector
analysis/          the noise study and data loading helpers
tests/             unit tests for collectors and analysis
deploy/            systemd service and timer files for the collection server
```

## How this was built

I built this with substantial help from Claude (Anthropic). Claude wrote most of the code in this repository, including the collectors, the tests and the analysis, and helped shape the research questions. The Ornn collector began from an existing starter script and was then reworked. I set up and run the collection server, ran every analysis on real data, and checked the results.

## License

The code is MIT licensed. This covers the code only, not any data collected with it.

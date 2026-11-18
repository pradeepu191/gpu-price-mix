"""Tests for the price-band logic, using a fake market instead of the real API."""

import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "collectors" / "vast"))

import collect_vast  # noqa: E402


def parse_band(query):
    """Read the lower and upper price limits back out of a query string."""
    lower = None
    upper = None
    for part in query.split():
        if part.startswith("dph_total>="):
            lower = float(part[len("dph_total>="):])
        if part.startswith("dph_total<"):
            upper = float(part[len("dph_total<"):])
    return lower, upper


def make_market(prices, cap):
    """A fake run_query that filters by price band, then returns at most `cap` offers
    in a shuffled order. This mimics a server whose ordering cannot be trusted."""
    offers = []
    for index, price in enumerate(prices):
        offers.append({"id": index, "dph_total": price})
    shuffler = random.Random(0)
    calls = []

    def run_query(market_type, query):
        calls.append(query)
        lower, upper = parse_band(query)
        matching = []
        for offer in offers:
            if lower is not None and offer["dph_total"] < lower:
                continue
            if upper is not None and offer["dph_total"] >= upper:
                continue
            matching.append(offer)
        shuffler.shuffle(matching)
        return 0, json.dumps(matching[:cap]).encode(), ""

    return run_query, calls


class BandTests(unittest.TestCase):
    def test_small_market_is_read_completely(self):
        run_query, calls = make_market([0.1, 0.6, 3.0, 70.0], cap=4)
        pages, summary = collect_vast.read_market(run_query, "on-demand", page_cap=4)
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["unique_offers"], 4)

    def test_crowded_band_is_split_until_complete(self):
        generator = random.Random(1)
        prices = []
        for _ in range(300):
            prices.append(round(generator.uniform(0.0, 0.25), 4))
        run_query, calls = make_market(prices, cap=16)
        pages, summary = collect_vast.read_market(run_query, "on-demand", page_cap=16)
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["unique_offers"], len(prices))

    def test_open_ended_top_band_is_split(self):
        prices = []
        for index in range(40):
            prices.append(70.0 + index)
        run_query, calls = make_market(prices, cap=8)
        pages, summary = collect_vast.read_market(run_query, "on-demand", page_cap=8)
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["unique_offers"], 40)

    def test_untrusted_order_does_not_lose_offers(self):
        # The bug seen in Nov 2026: a capped response is not the cheapest offers in the band
        generator = random.Random(2)
        prices = []
        for _ in range(1000):
            prices.append(round(generator.uniform(0.0, 10.0), 4))
        run_query, calls = make_market(prices, cap=50)
        pages, summary = collect_vast.read_market(run_query, "on-demand", page_cap=50)
        self.assertEqual(summary["unique_offers"], len(set(range(1000))))

    def test_identical_prices_beyond_cap_are_flagged_not_hidden(self):
        prices = [1.5] * 10
        run_query, calls = make_market(prices, cap=4)
        pages, summary = collect_vast.read_market(run_query, "on-demand", page_cap=4)
        self.assertEqual(summary["status"], "incomplete")
        problems = []
        for page in pages:
            problems.append(page["problem"])
        self.assertIn("tie_overflow", problems)

    def test_cli_failure_is_recorded_and_archived(self):
        def run_query(market_type, query):
            return 1, b"", "boom"
        pages, summary = collect_vast.read_market(run_query, "bid", page_cap=4)
        self.assertEqual(summary["status"], "failed")
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            collect_vast.archive_market(root, "RUN", "2026-11-18", "bid", pages, summary)
            page_rows = (root / "manifest" / "vast_pages.jsonl").read_text().splitlines()
            run_rows = (root / "manifest" / "vast_runs.jsonl").read_text().splitlines()
            self.assertEqual(json.loads(page_rows[0])["problem"], "cli_error")
            self.assertEqual(json.loads(run_rows[0])["status"], "failed")

    def test_band_edges_do_not_overlap(self):
        halves = collect_vast.split_band(1.0, 2.0)
        self.assertEqual(halves[0][1], halves[1][0])
        self.assertEqual(collect_vast.build_query(1.0, 1.5), "rentable=any dph_total>=1.000000 dph_total<1.500000")


if __name__ == "__main__":
    unittest.main()

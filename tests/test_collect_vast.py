"""Tests for the paging logic, using a fake market instead of the real API."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "collectors" / "vast"))

import collect_vast  # noqa: E402


def make_market(prices):
    """A fake run_query that behaves like Vast: filter by min price, sort ascending, cap results."""
    offers = []
    for index, price in enumerate(prices):
        offers.append({"id": index, "dph_total": price})

    calls = []

    def run_query(market_type, query, cap=4):
        calls.append(query)
        lower = None
        if "dph_total>=" in query:
            lower = float(query.split("dph_total>=")[1])
        matching = []
        for offer in offers:
            if lower is None or offer["dph_total"] >= lower:
                matching.append(offer)
        matching.sort(key=lambda offer: offer["dph_total"])
        return 0, json.dumps(matching[:cap]).encode(), ""

    return run_query, calls


class PagingTests(unittest.TestCase):
    def test_small_market_is_one_page(self):
        run_query, calls = make_market([0.5, 1.0, 2.0])
        pages, summary = collect_vast.read_market(run_query, "on-demand", page_cap=4)
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["unique_offers"], 3)
        self.assertEqual(len(calls), 1)

    def test_large_market_pages_and_dedupes(self):
        prices = [0.1, 0.2, 0.3, 0.4, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        run_query, calls = make_market(prices)
        pages, summary = collect_vast.read_market(run_query, "on-demand", page_cap=4)
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["unique_offers"], len(prices))
        self.assertGreater(len(calls), 1)

    def test_ties_beyond_cap_are_flagged_not_hidden(self):
        prices = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0]
        run_query, calls = make_market(prices)
        pages, summary = collect_vast.read_market(run_query, "on-demand", page_cap=4)
        self.assertEqual(summary["status"], "incomplete")
        self.assertEqual(pages[-1]["problem"], "tie_overflow")

    def test_unsorted_server_response_is_flagged(self):
        def run_query(market_type, query):
            offers = [{"id": 1, "dph_total": 2.0}, {"id": 2, "dph_total": 1.0}]
            return 0, json.dumps(offers).encode(), ""
        pages, summary = collect_vast.read_market(run_query, "on-demand", page_cap=4)
        self.assertEqual(summary["status"], "incomplete")
        self.assertEqual(pages[-1]["problem"], "unsorted")

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

    def test_floor_never_rounds_up(self):
        self.assertLessEqual(collect_vast.floor_price(2.4736111111111114), 2.4736111111111114)


if __name__ == "__main__":
    unittest.main()

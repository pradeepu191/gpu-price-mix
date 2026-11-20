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
        offers.append({"id": index, "dph_total": price, "machine_id": index, "gpu_name": "H100 SXM"})
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
        pages, summary, _machines = collect_vast.read_market(run_query, "on-demand", models=["H100 SXM"], page_cap=4)
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["unique_offers"], 4)

    def test_crowded_band_is_split_until_complete(self):
        generator = random.Random(1)
        prices = []
        for _ in range(300):
            prices.append(round(generator.uniform(0.0, 0.25), 4))
        run_query, calls = make_market(prices, cap=16)
        pages, summary, _machines = collect_vast.read_market(run_query, "on-demand", models=["H100 SXM"], page_cap=16)
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["unique_offers"], len(prices))

    def test_open_ended_top_band_is_split(self):
        prices = []
        for index in range(40):
            prices.append(70.0 + index)
        run_query, calls = make_market(prices, cap=8)
        pages, summary, _machines = collect_vast.read_market(run_query, "on-demand", models=["H100 SXM"], page_cap=8)
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["unique_offers"], 40)

    def test_untrusted_order_does_not_lose_offers(self):
        # The bug seen in Nov 2026: a capped response is not the cheapest offers in the band
        generator = random.Random(2)
        prices = []
        for _ in range(1000):
            prices.append(round(generator.uniform(0.0, 10.0), 4))
        run_query, calls = make_market(prices, cap=50)
        pages, summary, _machines = collect_vast.read_market(run_query, "on-demand", models=["H100 SXM"], page_cap=50)
        self.assertEqual(summary["unique_offers"], len(set(range(1000))))

    def test_identical_prices_beyond_cap_are_flagged_not_hidden(self):
        prices = [1.5] * 10
        run_query, calls = make_market(prices, cap=4)
        pages, summary, _machines = collect_vast.read_market(run_query, "on-demand", models=["H100 SXM"], page_cap=4)
        self.assertEqual(summary["status"], "incomplete")
        problems = []
        for page in pages:
            problems.append(page["problem"])
        self.assertIn("tie_overflow", problems)

    def test_cli_failure_is_recorded_and_archived(self):
        def run_query(market_type, query):
            return 1, b"", "boom"
        pages, summary, _machines = collect_vast.read_market(run_query, "bid", models=["H100 SXM"], page_cap=4)
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
        self.assertEqual(
            collect_vast.build_query("H100 SXM", 1.0, 1.5),
            "rentable=any gpu_name=H100_SXM dph_total>=1.000000 dph_total<1.500000",
        )
        self.assertEqual(collect_vast.build_query("H100 SXM", None, None), "rentable=any gpu_name=H100_SXM")

    def test_quota_error_stops_the_read(self):
        calls = []
        def run_query(market_type, query):
            calls.append(query)
            body = {"error": True, "status_code": 429, "msg": "Daily search row quota exhausted."}
            return 0, json.dumps(body).encode(), ""
        pages, summary, _machines = collect_vast.read_market(run_query, "on-demand", models=["H100 SXM", "B200"], page_cap=4)
        self.assertEqual(summary["status"], "quota_exhausted")
        self.assertEqual(len(calls), 1)
        self.assertEqual(pages[0]["problem"], "quota_exhausted")

    def test_rows_returned_counts_every_row_fetched(self):
        prices = []
        for index in range(10):
            prices.append(0.5 + index)
        run_query, calls = make_market(prices, cap=4)
        pages, summary, _machines = collect_vast.read_market(run_query, "on-demand", models=["H100 SXM"], page_cap=4)
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["unique_offers"], 10)
        self.assertGreaterEqual(summary["rows_returned"], 10)


class LookupTests(unittest.TestCase):
    def setUp(self):
        self.now = collect_vast.utc_now()
        self.calls = []

    def fake(self, live):
        """run_query that answers machine_id queries from the dict `live` (id -> offers)."""
        def run_query(market_type, query):
            self.calls.append(query)
            machine_id = query.split("machine_id=")[1]
            reply = live.get(machine_id, [])
            return 0, json.dumps(reply).encode(), ""
        return run_query

    def roster(self, ids, days_ago=0):
        stamp = (self.now - collect_vast.timedelta(days=days_ago)).isoformat()
        result = {}
        for machine_id in ids:
            result[machine_id] = {"gpu_name": "L40S", "last_seen_utc": stamp}
        return result

    def test_hidden_live_machine_is_found(self):
        live = {"7": [{"id": 70, "machine_id": 7, "gpu_name": "L40S"}]}
        seen = {"5": "L40S"}
        pages, info = collect_vast.look_up_missing(
            self.fake(live), "on-demand", self.roster(["5", "7", "9"]), seen, 3, self.now, sleep=lambda s: None)
        self.assertEqual(info["lookups"], 2)          # 5 was already seen, so only 7 and 9 are asked
        self.assertEqual(info["lookups_live"], 1)
        self.assertIn("7", seen)
        self.assertNotIn("9", seen)
        self.assertEqual(pages[0]["page"], 3)
        self.assertEqual(pages[0]["kind"], "machine_lookup")

    def test_machines_unseen_for_over_a_week_are_not_asked(self):
        pages, info = collect_vast.look_up_missing(
            self.fake({}), "on-demand", self.roster(["1"], days_ago=8), {}, 0, self.now, sleep=lambda s: None)
        self.assertEqual(info["lookups"], 0)

    def test_quota_error_stops_lookups(self):
        def run_query(market_type, query):
            self.calls.append(query)
            body = {"error": True, "status_code": 429, "msg": "Daily search row quota exhausted."}
            return 0, json.dumps(body).encode(), ""
        pages, info = collect_vast.look_up_missing(
            run_query, "on-demand", self.roster(["1", "2", "3"]), {}, 0, self.now, sleep=lambda s: None)
        self.assertEqual(info["lookup_status"], "quota_exhausted")
        self.assertEqual(len(self.calls), 1)

    def test_rate_limit_is_retried_not_treated_as_quota(self):
        replies = [{"error": True, "status_code": 429, "msg": "API requests too frequent"},
                   [{"id": 1, "machine_id": 1, "gpu_name": "L40S"}]]
        def run_query(market_type, query):
            return 0, json.dumps(replies.pop(0)).encode(), ""
        seen = {}
        pages, info = collect_vast.look_up_missing(
            run_query, "on-demand", self.roster(["1"]), seen, 0, self.now, sleep=lambda s: None)
        self.assertEqual(info["lookup_status"], "complete")
        self.assertEqual(info["lookups_live"], 1)

    def test_roster_saves_seen_and_drops_old(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            roster = self.roster(["old"], days_ago=9)
            roster.update(self.roster(["recent"], days_ago=2))
            kept = collect_vast.save_roster(root, "on-demand", roster, {"new": "B200"}, self.now)
            self.assertEqual(sorted(kept), ["new", "recent"])
            self.assertEqual(sorted(collect_vast.load_roster(root, "on-demand")), ["new", "recent"])

    def test_bootstrap_reads_saved_snapshots(self):
        import gzip
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_id = self.now.strftime("%Y%m%dT%H%M%SZ")
            run_dir = root / "raw" / "vast" / "type=on-demand" / f"dt={self.now.strftime('%Y-%m-%d')}" / f"run={run_id}"
            run_dir.mkdir(parents=True)
            with gzip.open(run_dir / "page_000.json.gz", "wb") as handle:
                handle.write(json.dumps([{"id": 1, "machine_id": 42, "gpu_name": "H200"}]).encode())
            with gzip.open(run_dir / "page_001.json.gz", "wb") as handle:
                handle.write(json.dumps({"error": True}).encode())
            roster = collect_vast.bootstrap_roster(root, "on-demand", self.now)
            self.assertEqual(list(roster), ["42"])
            self.assertEqual(roster["42"]["gpu_name"], "H200")


if __name__ == "__main__":
    unittest.main()

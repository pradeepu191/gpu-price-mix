"""Tests for the probe that checks Computable's Vast query for missed machines.

The fixture replies are made up. They have the shape of Vast's replies but no
real machines, sellers or prices.
"""

import gzip
import json
import sys
import tempfile
import unittest
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

import computable_probe as probe  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "computable_probe"
CHIP = "H100 SXM"


def fixture(name):
    return json.loads((FIXTURES / name).read_text())


class TheirLogicTests(unittest.TestCase):
    def test_query_matches_their_collector(self):
        self.assertEqual(probe.their_query("H100 SXM"), {
            "gpu_name": {"eq": "H100 SXM"}, "rentable": {"eq": True}, "type": "on-demand",
            "order": [["dph_total", "asc"]], "limit": 50})
        url = probe.their_url(probe.their_query("H100 SXM", "desc"))
        self.assertTrue(url.startswith("https://console.vast.ai/api/v0/bundles/?q=%7B"))
        sent = json.loads(urllib.parse.unquote(url.split("?q=", 1)[1]))
        self.assertEqual(sent["order"], [["dph_total", "desc"]])

    def test_asking_for_one_machine_only_adds_the_machine_id(self):
        plain = probe.their_query("B200")
        one = probe.their_query("B200", machine_id="123")
        self.assertEqual(one.pop("machine_id"), {"eq": 123})
        self.assertEqual(one, plain)

    def test_parser_drops_what_theirs_drops(self):
        kept, skipped = probe.parse_their_offers(fixture("their_before.json")["offers"], CHIP)
        self.assertEqual(sorted({row["machine_id"] for row in kept}), [101, 102, 103])
        self.assertEqual(skipped, 3)   # a bid, another GPU model, and a zero GPU count

    def test_one_row_per_machine_cheapest_per_gpu(self):
        kept, _ = probe.parse_their_offers(fixture("their_before.json")["offers"], CHIP)
        ranked = probe.rank_machines(kept)
        self.assertEqual([row["machine_id"] for row in ranked], [103, 102, 101])
        row_101 = ranked[-1]
        self.assertAlmostEqual(row_101["per_gpu"], 2.40)   # the 4-GPU offer, not the $2.50 single
        self.assertEqual(row_101["num_gpus"], 4)

    def test_verified_us_ca_screen(self):
        self.assertTrue(probe.is_verified_us_ca("verified", "Oregon, US"))
        self.assertTrue(probe.is_verified_us_ca("verified", "Quebec, CA"))
        self.assertTrue(probe.is_verified_us_ca("verified", ", US"))
        self.assertFalse(probe.is_verified_us_ca("deverified", "Oregon, US"))
        self.assertFalse(probe.is_verified_us_ca("unverified", "Oregon, US"))
        self.assertFalse(probe.is_verified_us_ca("verified", "Taiwan, TW"))
        self.assertFalse(probe.is_verified_us_ca("verified", None))

    def test_weighted_median(self):
        self.assertAlmostEqual(probe.weighted_median([(2.0, 2), (2.4, 4)]), 2.4)
        self.assertAlmostEqual(probe.weighted_median([(1.0, 1), (2.0, 1), (3.0, 1)]), 2.0)
        # The running weight lands exactly on half: average of the two values around it.
        self.assertAlmostEqual(probe.weighted_median([(1.0, 1), (3.0, 1)]), 2.0)
        self.assertAlmostEqual(probe.weighted_median([(1.7, 8), (1.8, 2), (2.0, 2), (2.4, 4)]), 1.75)
        with self.assertRaises(ValueError):
            probe.weighted_median([])

    def test_floors_hold_a_thin_book_out(self):
        rows = [{"machine_id": i, "host_id": i % 2, "per_gpu": 2.0 + i / 10, "num_gpus": 1,
                 "verification": "verified", "geolocation": "Ohio, US"} for i in range(5)]
        self.assertEqual(probe.seat_statistic(rows, 5, 3)["status"], "held out (thin_book)")   # 2 hosts only
        rows[0]["host_id"] = 7
        priced = probe.seat_statistic(rows, 5, 3)
        self.assertEqual(priced["status"], "priced")
        self.assertAlmostEqual(priced["median"], 2.2)
        self.assertEqual(probe.seat_statistic(rows[:2], None, None)["status"], "priced")   # no floors on B200
        self.assertEqual(probe.seat_statistic([], 5, 3)["status"], "no eligible machines")

    def test_descending_call_is_merged_and_flags_a_gap(self):
        asc = [{"id": i, "machine_id": i, "host_id": 1, "gpu_name": CHIP, "num_gpus": 1, "dph_total": 1.0 + i,
                "verification": "verified", "geolocation": "Ohio, US"} for i in range(50)]
        desc = [dict(asc[49]), dict(asc[49], id=900, machine_id=900)]
        book = probe.their_book(asc, desc, CHIP)
        self.assertTrue(book["fetch_truncated"])
        self.assertFalse(book["coverage_gap"])
        self.assertEqual(len(book["machines"]), 51)
        self.assertTrue(probe.their_book(asc, asc, CHIP)["coverage_gap"])
        self.assertFalse(probe.their_book(asc[:10], None, CHIP)["fetch_truncated"])


class CompareTests(unittest.TestCase):
    def setUp(self):
        self.before = probe.their_book(fixture("their_before.json")["offers"], None, CHIP)
        self.after = probe.their_book(fixture("their_after.json")["offers"], None, CHIP)
        self.wide = probe.known_machines(fixture("mine.json"), CHIP)
        self.archive = probe.known_machines(fixture("archive.json"), CHIP)
        by_machine = fixture("by_machine.json")
        self.direct = {machine_id: reply["offers"] for machine_id, reply in by_machine.items() if machine_id != "101"}
        # Asked again after their second run: 105 has been rented in the meantime.
        self.again = {"104": by_machine["104"]["offers"], "105": [], "106": by_machine["106"]["offers"],
                      "109": by_machine["109"]["offers"]}

    def labels(self):
        found = probe.compare_chip(CHIP, self.before, self.after, self.direct, self.wide, self.archive,
                                   set(self.direct), self.again)
        return {label["machine_id"]: label for label in found}

    def test_missing_machines_are_the_ones_not_in_their_first_result(self):
        self.assertEqual(sorted(self.labels()), ["104", "105", "106", "107", "108", "109"])

    def test_labels(self):
        labels = self.labels()
        self.assertTrue(labels["104"]["missed"])
        self.assertTrue(labels["104"]["passes_their_filters"])
        self.assertAlmostEqual(labels["104"]["per_gpu"], 1.80)
        self.assertTrue(labels["104"]["in_my_query"])
        # 106 was hidden from my own query too, and known only from the archive.
        self.assertTrue(labels["106"]["missed"])
        self.assertFalse(labels["106"]["in_my_query"])
        self.assertEqual(labels["106"]["gpu_count"], 8)
        # 105 answered once, then was gone: not counted.
        self.assertTrue(labels["105"]["seen_once"])
        self.assertFalse(labels["105"]["missed"])
        # 108 showed up in their second run: it became rentable mid-run, not counted.
        self.assertTrue(labels["108"]["rentable_now"])
        self.assertFalse(labels["108"]["missed"])
        # 107 has no rentable offer.
        self.assertFalse(labels["107"]["rentable_now"])
        self.assertFalse(labels["107"]["verified"])
        # 109 is missed but "deverified", so it does not pass their screen.
        self.assertTrue(labels["109"]["missed"])
        self.assertFalse(labels["109"]["passes_their_filters"])

    def test_a_machine_not_asked_about_stays_unknown(self):
        del self.direct["106"]
        label = self.labels()["106"]
        self.assertIsNone(label["rentable_now"])
        self.assertFalse(label["missed"])

    def test_seat_with_and_without_the_missed_machines(self):
        chip = {"before": self.before, "labels": list(self.labels().values())}
        no_floors = probe.seat_report({"min_machines": None, "min_hosts": None}, [chip])
        self.assertAlmostEqual(no_floors["as their query sees it"]["median"], 2.40)
        self.assertAlmostEqual(no_floors["with the missed machines"]["median"], 1.75)
        self.assertEqual(no_floors["with the missed machines"]["eligible_machines"], 4)   # 109 is screened out
        self.assertTrue(no_floors["changed"])
        floors = probe.seat_report({"min_machines": 5, "min_hosts": 3}, [chip])
        self.assertEqual(floors["as their query sees it"]["status"], "held out (thin_book)")
        self.assertEqual(floors["with the missed machines"]["status"], "held out (thin_book)")
        # Held out either way: the counts move, but what the seat would publish does not.
        self.assertFalse(floors["changed"])
        self.assertTrue(floors["counts_changed"])

    def test_lookup_order_puts_likely_misses_first(self):
        order = probe.lookup_order({"104", "105", "106", "107", "108", "109"}, self.wide, self.archive)
        self.assertEqual(order[:3], ["104", "105", "109"])       # rentable in my own query
        self.assertEqual(order[3:5], ["106", "108"])             # hidden from my query, rentable in the archive
        self.assertEqual(order[5], "107")


class EndToEndTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.now = datetime(2026, 11, 26, 15, 30, tzinfo=timezone.utc)
        run_id = "20261126T140210Z"
        raw = self.root / "raw" / "vast" / "type=on-demand" / "dt=2026-11-26" / f"run={run_id}"
        raw.mkdir(parents=True)
        with gzip.open(raw / "page_000.json.gz", "wb") as handle:
            handle.write((FIXTURES / "archive.json").read_bytes())
        (self.root / "manifest").mkdir()
        rows = [{"run_id": run_id, "market_type": "on-demand", "status": "complete"},
                {"run_id": "20261120T140210Z", "market_type": "on-demand", "status": "complete"}]   # too old
        (self.root / "manifest" / "vast_runs.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        self.by_machine = fixture("by_machine.json")
        self.asked = []
        self.their_calls = 0

    def tearDown(self):
        self.tmp.cleanup()

    def get(self, url):
        query = json.loads(urllib.parse.unquote(url.split("?q=", 1)[1]))
        self.asked.append(query)
        if "machine_id" in query:
            machine_id = str(query["machine_id"]["eq"])
            times = sum(1 for q in self.asked if q.get("machine_id") == query["machine_id"])
            if machine_id == "105" and times > 1:
                return 200, b'{"offers": []}', None
            return 200, json.dumps(self.by_machine[machine_id]).encode(), None
        self.their_calls += 1
        name = "their_before.json" if self.their_calls == 1 else "their_after.json"
        return 200, (FIXTURES / name).read_bytes(), None

    def cli(self, query):
        self.cli_query = query
        return 0, (FIXTURES / "mine.json").read_bytes(), ""

    def run_it(self, **kwargs):
        return probe.run_probe(self.root, ["h100_sxm"], kwargs.pop("get", self.get), kwargs.pop("cli", self.cli),
                               self.now, sleep=lambda seconds: None, **kwargs)

    def test_full_run(self):
        run = self.run_it()
        chip = run["chips"][0]
        self.assertIsNone(run["stopped"])
        self.assertEqual(self.cli_query, "rentable=true gpu_name=H100_SXM")
        self.assertEqual(run["archive_snapshots_used"], 1)
        self.assertEqual(chip["their_machines"], 3)
        self.assertEqual(chip["missing_from_theirs"], 6)
        self.assertEqual(chip["missed_rentable"], 3)
        self.assertEqual(chip["missed_passing_their_filters"], 2)
        self.assertEqual(sorted(chip["missed_machine_ids"]), ["104", "106", "109"])
        self.assertEqual(chip["not_rentable_now"], 1)
        self.assertEqual(chip["changed_during_run"], 2)
        self.assertEqual(run["seats"]["h100_sxm"]["as their query sees it"]["status"], "held out (thin_book)")
        # before, control, 6 by id, after, 4 asked again = 13 without a key. 1 with my key.
        self.assertEqual(run["calls"], {"without a key": 13, "with my key": 1})
        self.assertEqual(run["key_rows_used"], 7)

    def test_every_request_is_logged_and_saved(self):
        run = self.run_it()
        log = (self.root / "manifest" / "computable_probe_requests.jsonl").read_text().splitlines()
        self.assertEqual(len(log), run["calls_total"])
        first = json.loads(log[0])
        self.assertEqual(first["label"], "H100_SXM_theirs_before_asc")
        self.assertEqual(first["transport"], "without a key")
        saved = self.root / first["raw_file"]
        with gzip.open(saved, "rb") as handle:
            self.assertEqual(handle.read(), (FIXTURES / "their_before.json").read_bytes())
        runs = (self.root / "manifest" / "computable_probe_runs.jsonl").read_text().splitlines()
        self.assertEqual(len(runs), 1)
        self.assertNotIn("details", json.loads(runs[0]))
        # The recorder's own files are untouched.
        self.assertEqual(len((self.root / "manifest" / "vast_runs.jsonl").read_text().splitlines()), 2)
        self.assertFalse((self.root / "state").exists())

    def test_the_lookup_cap_is_respected(self):
        run = self.run_it(max_lookups=2)
        self.assertEqual(run["chips"][0]["looked_up"], 2)
        self.assertEqual(run["chips"][0]["not_looked_up"], 4)

    def test_a_429_stops_the_run_at_once(self):
        def get(url):
            if "machine_id" in urllib.parse.unquote(url) and len(self.asked) >= 3:
                return 429, b'{"error": "slow down"}', "HTTP 429"
            return self.get(url)
        run = self.run_it(get=get)
        self.assertIn("429", run["stopped"])
        self.assertEqual(run["chips"], [])

    def test_a_failed_control_stops_the_run(self):
        self.by_machine["101"] = {"offers": []}
        run = self.run_it()
        self.assertIn("control failed", run["stopped"])

    def test_a_used_up_quota_stops_the_run(self):
        def cli(query):
            return 1, json.dumps({"error": True, "status_code": 429,
                                  "msg": "Daily search row quota exhausted."}).encode(), ""
        run = self.run_it(cli=cli)
        self.assertIn("quota", run["stopped"])

    def test_recorder_window(self):
        self.assertTrue(probe.inside_recorder_window(datetime(2026, 11, 26, 14, 5, tzinfo=timezone.utc)))
        self.assertFalse(probe.inside_recorder_window(datetime(2026, 11, 26, 14, 20, tzinfo=timezone.utc)))
        self.assertFalse(probe.inside_recorder_window(datetime(2026, 11, 26, 15, 5, tzinfo=timezone.utc)))


if __name__ == "__main__":
    unittest.main()

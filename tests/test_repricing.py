"""Tests for spotting asking prices that move while a card cannot be rented."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

import repricing  # noqa: E402


def card(gpu_id, price, free=False, model="H100 SXM", machine_id=1, host_id=1):
    return {"gpu_id": gpu_id, "gpu_name": model, "machine_id": machine_id, "host_id": host_id,
            "price": price, "free": free, "occupied": not free}


class FindRepricersTests(unittest.TestCase):
    def test_flags_a_card_that_moved_while_not_rentable(self):
        runs = {"r1": [card("a", 4.0)], "r2": [card("a", 6.0)]}
        flagged = repricing.find_repricers(runs, 0.20)
        self.assertIn(("H100 SXM", "a"), flagged)
        self.assertEqual(flagged[("H100 SXM", "a")]["changes"], 1)
        self.assertAlmostEqual(flagged[("H100 SXM", "a")]["low"], 4.0)
        self.assertAlmostEqual(flagged[("H100 SXM", "a")]["high"], 6.0)

    def test_small_moves_are_not_flagged(self):
        runs = {"r1": [card("a", 4.0)], "r2": [card("a", 4.5)]}
        self.assertEqual(repricing.find_repricers(runs, 0.20), {})

    def test_readings_where_the_card_was_rentable_are_ignored(self):
        # The only big move happens while the card can be rented, so it is a real offer.
        runs = {"r1": [card("a", 4.0)], "r2": [card("a", 6.0, free=True)], "r3": [card("a", 4.0)]}
        self.assertEqual(repricing.find_repricers(runs, 0.20), {})

    def test_a_card_seen_once_cannot_be_flagged(self):
        runs = {"r1": [card("a", 4.0)], "r2": [card("b", 9.0)]}
        self.assertEqual(repricing.find_repricers(runs, 0.20), {})

    def test_same_id_under_two_models_is_two_cards(self):
        runs = {"r1": [card("a", 4.0), card("a", 2.0, model="L40S")],
                "r2": [card("a", 6.0), card("a", 2.0, model="L40S")]}
        flagged = repricing.find_repricers(runs, 0.20)
        self.assertEqual(list(flagged), [("H100 SXM", "a")])

    def test_counts_every_price_change(self):
        runs = {"r1": [card("a", 4.0)], "r2": [card("a", 6.0)], "r3": [card("a", 6.0)], "r4": [card("a", 4.0)]}
        self.assertEqual(repricing.find_repricers(runs, 0.20)[("H100 SXM", "a")]["changes"], 2)


class ThreeWaysTests(unittest.TestCase):
    def test_split(self):
        cards = [card("a", 6.0), card("b", 2.0), card("c", 2.5, free=True)]
        flagged = {("H100 SXM", "a"): {}}
        ways = repricing.three_ways(cards, flagged)
        self.assertEqual(len(ways["all cards"]), 3)
        self.assertEqual([c["gpu_id"] for c in ways["without those cards"]], ["b", "c"])
        self.assertEqual([c["gpu_id"] for c in ways["rentable cards only"]], ["c"])

    def test_average_and_swing(self):
        self.assertAlmostEqual(repricing.average_price([card("a", 2.0), card("b", 4.0)]), 3.0)
        self.assertAlmostEqual(repricing.swing([2.0, 2.5, 2.2]), 0.25)


class SummaryTests(unittest.TestCase):
    def make_runs(self):
        steady = []
        for index in range(12):
            steady.append((f"s{index}", 2.0 + 0.05 * index))
        runs = {}
        for run_id, mover_price in (("r1", 4.0), ("r2", 6.0), ("r3", 4.0)):
            cards = []
            for gpu_id, price in steady:
                cards.append(card(gpu_id, price, free=gpu_id in ("s0", "s1"), machine_id=2, host_id=2))
            for index in range(4):
                cards.append(card(f"m{index}", mover_price, machine_id=9, host_id=9))
            runs[run_id] = cards
        return runs

    def test_removing_the_movers_removes_the_swing(self):
        runs = self.make_runs()
        flagged = repricing.find_repricers(runs, 0.20)
        self.assertEqual(len(flagged), 4)
        rows = repricing.snapshot_rows("H100 SXM", runs, flagged, 200, 7)
        summary = repricing.summarize("H100 SXM", rows, flagged)
        self.assertEqual(summary["snapshots"], 3)
        self.assertEqual(summary["flagged machines"], 1)
        self.assertEqual(summary["flagged sellers"], 1)
        self.assertAlmostEqual(summary["flagged share of supply"], 4 / 16)
        self.assertGreater(summary["average swing: all cards"], 0.10)
        self.assertAlmostEqual(summary["average swing: without those cards"], 0.0)
        # Only two rentable cards, too few for a chance band, but the average still exists.
        self.assertIsNone(summary["band low median: rentable cards only"])
        self.assertAlmostEqual(summary["average low: rentable cards only"], 2.025)

    def test_other_models_are_left_out(self):
        runs = self.make_runs()
        runs["r1"].append(card("z", 1.0, model="L40S"))
        rows = repricing.snapshot_rows("H100 SXM", runs, {}, 50, 7)
        self.assertEqual(rows[0]["cards: all cards"], 16)


if __name__ == "__main__":
    unittest.main()

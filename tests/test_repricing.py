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


class ControlTests(unittest.TestCase):
    def test_rentable_movers_are_the_control_group(self):
        runs = {"r1": [card("a", 2.0, free=True), card("b", 4.0)],
                "r2": [card("a", 3.0, free=True), card("b", 6.0)]}
        self.assertEqual(list(repricing.find_movers(runs, 0.20, while_rentable=True)), [("H100 SXM", "a")])
        self.assertEqual(list(repricing.find_repricers(runs, 0.20)), [("H100 SXM", "b")])

    def test_control_set_drops_only_control_cards(self):
        cards = [card("a", 6.0), card("b", 2.0), card("c", 2.5, free=True)]
        ways = repricing.three_ways(cards, {("H100 SXM", "a"): {}}, {("H100 SXM", "c"): {}})
        self.assertEqual([c["gpu_id"] for c in ways["control"]], ["a", "b"])
        # With no control group given, the control set is every card.
        self.assertEqual(len(repricing.three_ways(cards, {})["control"]), 3)


class SellerOverlapTests(unittest.TestCase):
    def test_counts_sellers_and_models(self):
        flagged = {
            ("H100 SXM", "a"): {"host_id": 1}, ("H100 SXM", "b"): {"host_id": 1},
            ("H200", "c"): {"host_id": 1}, ("H200", "d"): {"host_id": 2},
        }
        overlap = repricing.seller_overlap(flagged)
        self.assertEqual(overlap["sellers"], 2)
        self.assertEqual(overlap["sellers in two or more models"], 1)
        self.assertAlmostEqual(overlap["largest seller share"], 0.75)
        self.assertAlmostEqual(overlap["top three sellers share"], 1.0)

    def test_empty(self):
        self.assertEqual(repricing.seller_overlap({})["sellers"], 0)


class DecomposeTests(unittest.TestCase):
    def parts(self, step):
        return (step["price moved, not rentable"], step["price moved, rentable"], step["entering or leaving"])

    def test_repricing_a_card_nobody_can_rent(self):
        runs = {"r1": [card("a", 4.0), card("b", 2.0)], "r2": [card("a", 6.0), card("b", 2.0)]}
        step = repricing.decompose_steps("H100 SXM", runs)[0]
        self.assertAlmostEqual(step["change"], 1.0)
        not_rentable, rentable, mix = self.parts(step)
        self.assertAlmostEqual(not_rentable, 1.0)
        self.assertAlmostEqual(rentable, 0.0)
        self.assertAlmostEqual(mix, 0.0)

    def test_repricing_a_rentable_card(self):
        runs = {"r1": [card("a", 4.0, free=True), card("b", 2.0)], "r2": [card("a", 6.0), card("b", 2.0)]}
        not_rentable, rentable, mix = self.parts(repricing.decompose_steps("H100 SXM", runs)[0])
        self.assertAlmostEqual(not_rentable, 0.0)
        self.assertAlmostEqual(rentable, 1.0)
        self.assertAlmostEqual(mix, 0.0)

    def test_a_card_entering_is_mix(self):
        runs = {"r1": [card("a", 2.0), card("b", 2.0)], "r2": [card("a", 2.0), card("b", 2.0), card("c", 5.0)]}
        step = repricing.decompose_steps("H100 SXM", runs)[0]
        not_rentable, rentable, mix = self.parts(step)
        self.assertAlmostEqual(step["change"], 1.0)
        self.assertAlmostEqual(not_rentable, 0.0)
        self.assertAlmostEqual(mix, 1.0)
        self.assertEqual((step["cards staying"], step["cards entered"], step["cards left"]), (2, 1, 0))

    def test_parts_always_add_up_to_the_change(self):
        runs = {
            "r1": [card("a", 4.0), card("b", 2.0, free=True), card("c", 3.0)],
            "r2": [card("a", 6.5), card("b", 2.4), card("d", 9.0), card("z", 1.0, model="L40S")],
            "r3": [card("a", 3.9), card("d", 9.0, free=True)],
        }
        steps = repricing.decompose_steps("H100 SXM", runs)
        self.assertEqual(len(steps), 2)
        for step in steps:
            self.assertAlmostEqual(sum(self.parts(step)), step["change"])
            self.assertAlmostEqual(step["change"], step["average now"] - step["average before"])

    def test_step_summary_is_a_share_of_the_price_level(self):
        runs = {"r1": [card("a", 4.0), card("b", 2.0)], "r2": [card("a", 6.0), card("b", 2.0)]}
        summary = repricing.summarize_steps(repricing.decompose_steps("H100 SXM", runs))
        self.assertAlmostEqual(summary["change"], 1.0 / 3.0)
        self.assertAlmostEqual(summary["price moved, not rentable"], 1.0 / 3.0)
        self.assertAlmostEqual(summary["entering or leaving"], 0.0)


class GapFillTests(unittest.TestCase):
    def test_a_short_gap_is_filled_with_the_last_reading(self):
        runs = {"r1": [card("a", 2.0, free=True)], "r2": [], "r3": [card("a", 2.0)]}
        filled, counts = repricing.fill_short_gaps(runs, max_gap=3)
        self.assertEqual(len(filled["r2"]), 1)
        self.assertTrue(filled["r2"][0]["filled"])
        self.assertTrue(filled["r2"][0]["free"])
        self.assertAlmostEqual(filled["r2"][0]["price"], 2.0)
        self.assertEqual(counts["back within the limit"], 1)
        self.assertEqual(counts["same price on return"], 1)
        self.assertEqual(runs["r2"], [])   # the input is left alone

    def test_a_long_gap_is_not_filled(self):
        runs = {"r1": [card("a", 2.0)], "r2": [], "r3": [], "r4": [card("a", 2.0)]}
        filled, counts = repricing.fill_short_gaps(runs, max_gap=1)
        self.assertEqual(filled["r2"], [])
        self.assertEqual(counts["back later"], 1)

    def test_a_card_that_never_returns_is_not_filled(self):
        runs = {"r1": [card("a", 2.0), card("b", 3.0)], "r2": [card("b", 3.0)]}
        filled, counts = repricing.fill_short_gaps(runs)
        self.assertEqual(len(filled["r2"]), 1)
        self.assertEqual(counts["not back yet"], 1)

    def test_filling_removes_fake_mix_from_the_steps(self):
        runs = {"r1": [card("a", 2.0), card("b", 6.0)], "r2": [card("a", 2.0)], "r3": [card("a", 2.0), card("b", 6.0)]}
        raw = repricing.decompose_steps("H100 SXM", runs)
        self.assertAlmostEqual(raw[0]["entering or leaving"], -2.0)
        filled, _ = repricing.fill_short_gaps(runs)
        for step in repricing.decompose_steps("H100 SXM", filled):
            self.assertAlmostEqual(step["change"], 0.0)


class RentalStartTests(unittest.TestCase):
    def test_free_then_taken_is_a_start_at_the_free_price(self):
        runs = {"20261121T000000Z": [card("a", 2.0, free=True), card("b", 5.0)],
                "20261121T020000Z": [card("a", 2.5), card("b", 5.0)]}
        changes = repricing.state_changes("H100 SXM", runs)
        kinds = sorted(change["kind"] for change in changes)
        self.assertEqual(kinds, ["got taken", "stayed taken"])
        hours = repricing.hours_between("20261121T000000Z", "20261121T020000Z")
        self.assertAlmostEqual(hours, 2.0)
        summary = repricing.summarize_rentals("H100 SXM", changes, 3.5, hours)
        self.assertEqual(summary["rental starts"], 1)
        self.assertAlmostEqual(summary["card starts per hour"], 0.5)
        self.assertAlmostEqual(summary["average price locked in"], 2.0)
        self.assertAlmostEqual(summary["asking average above locked in"], 0.75)
        self.assertAlmostEqual(summary["drift per step: stayed taken"], 0.0)

    def test_filled_readings_never_count_as_starts(self):
        runs = {"r1": [card("a", 2.0, free=True)], "r2": [], "r3": [card("a", 2.0)]}
        filled, _ = repricing.fill_short_gaps(runs)
        self.assertEqual(repricing.state_changes("H100 SXM", filled), [])

    def test_no_starts(self):
        summary = repricing.summarize_rentals("H100 SXM", [], 3.0, 10.0)
        self.assertEqual(summary["rental starts"], 0)
        self.assertIsNone(summary["average price locked in"])


class TravelTests(unittest.TestCase):
    def test_flapping_travels_far_and_ends_where_it_began(self):
        runs = {"r1": [card("a", 4.0)], "r2": [card("a", 6.0)], "r3": [card("a", 4.0)]}
        summary = repricing.summarize_steps(repricing.decompose_steps("H100 SXM", runs))
        self.assertAlmostEqual(summary["taken cards: net"], 0.0)
        self.assertAlmostEqual(summary["taken cards: travel"], 4.0 / 5.0)


if __name__ == "__main__":
    unittest.main()

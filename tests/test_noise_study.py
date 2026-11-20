"""Tests for the index formula and the chance simulation."""

import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

import noise_study  # noqa: E402


class FormulaTests(unittest.TestCase):
    def test_constant_prices_give_that_price(self):
        rentals = [(2.5, 1), (2.5, 8), (2.5, 2)]
        for cutoff in (0.0, 0.05, 0.1):
            self.assertAlmostEqual(noise_study.winsorized_weighted_mean(rentals, cutoff), 2.5)

    def test_no_cutoff_is_plain_gpu_weighted_average(self):
        rentals = [(1.0, 1), (3.0, 3)]
        self.assertAlmostEqual(noise_study.winsorized_weighted_mean(rentals, 0.0), 2.5)

    def test_doubling_every_price_doubles_the_index(self):
        rentals = [(1.0, 1), (2.0, 2), (9.0, 1), (1.5, 4)]
        doubled = []
        for price, weight in rentals:
            doubled.append((price * 2, weight))
        for cutoff in (0.0, 0.1):
            self.assertAlmostEqual(
                noise_study.winsorized_weighted_mean(doubled, cutoff),
                2 * noise_study.winsorized_weighted_mean(rentals, cutoff),
            )

    def test_cutoff_limits_an_extreme_price(self):
        rentals = []
        for _ in range(19):
            rentals.append((2.0, 1))
        rentals.append((100.0, 1))
        plain = noise_study.winsorized_weighted_mean(rentals, 0.0)
        pulled_in = noise_study.winsorized_weighted_mean(rentals, 0.1)
        self.assertGreater(plain, 6.0)
        self.assertAlmostEqual(pulled_in, 2.0)

    def test_weighted_quantile_uses_gpu_counts(self):
        rentals = [(1.0, 1), (5.0, 9)]
        self.assertEqual(noise_study.weighted_quantile(rentals, 0.05), 1.0)
        self.assertEqual(noise_study.weighted_quantile(rentals, 0.5), 5.0)


class SmoothQuantileTests(unittest.TestCase):
    def test_smooth_quantile_stays_inside_the_price_range(self):
        rentals = [(1.0, 1), (2.0, 3), (4.0, 1)]
        for share in (0.0, 0.05, 0.3, 0.5, 0.9, 1.0):
            value = noise_study.weighted_quantile(rentals, share, "smooth")
            self.assertGreaterEqual(value, 1.0)
            self.assertLessEqual(value, 4.0)

    def test_smooth_quantile_rises_with_share(self):
        rentals = [(1.0, 2), (2.0, 1), (3.0, 5), (7.0, 1)]
        previous = None
        for step in range(11):
            value = noise_study.weighted_quantile(rentals, step / 10, "smooth")
            if previous is not None:
                self.assertGreaterEqual(value, previous)
            previous = value

    def test_smooth_quantile_of_equal_weights_is_the_middle(self):
        rentals = [(1.0, 1), (3.0, 1)]
        self.assertAlmostEqual(noise_study.weighted_quantile(rentals, 0.5, "smooth"), 2.0)

    def test_constant_prices_give_that_price_with_smooth_cutoffs(self):
        rentals = [(2.5, 1), (2.5, 8)]
        self.assertAlmostEqual(noise_study.winsorized_weighted_mean(rentals, 0.1, "smooth"), 2.5)


class SimulationTests(unittest.TestCase):
    def test_more_rentals_means_less_chance_error(self):
        rng = random.Random(0)
        rentals = []
        for _ in range(60):
            rentals.append((rng.uniform(1.5, 3.5), rng.choice([1, 1, 2, 8])))
        few = noise_study.simulate(rentals, 3, 0.05, 3000, random.Random(1))
        many = noise_study.simulate(rentals, 100, 0.05, 3000, random.Random(1))
        self.assertGreater(few["p95"] - few["p05"], many["p95"] - many["p05"])

    def test_identical_offers_give_zero_error(self):
        rentals = [(2.0, 1)] * 10
        result = noise_study.simulate(rentals, 3, 0.05, 200, random.Random(2))
        self.assertAlmostEqual(result["p05"], 0.0)
        self.assertAlmostEqual(result["p95"], 0.0)
        self.assertAlmostEqual(result["bias"], 0.0)


class CardTests(unittest.TestCase):
    def nested_offers(self):
        # One 4-GPU machine listed as a 4-GPU bundle (not rentable, one card taken)
        # and as a 1-GPU bundle of a spare card (rentable), plus a second machine.
        return [
            {"id": 1, "gpu_name": "H100 SXM", "machine_id": 10, "host_id": 7, "num_gpus": 4,
             "dph_base": 8.0, "rentable": False, "gpu_ids": [1, 2, 3, 4]},
            {"id": 2, "gpu_name": "H100 SXM", "machine_id": 10, "host_id": 7, "num_gpus": 1,
             "dph_base": 2.0, "rentable": True, "gpu_ids": [4]},
            {"id": 3, "gpu_name": "H100 SXM", "machine_id": 11, "host_id": 8, "num_gpus": 2,
             "dph_base": 6.0, "rentable": False, "gpu_ids": [5, 6]},
            {"id": 4, "gpu_name": "B200", "machine_id": 12, "host_id": 8, "num_gpus": 1,
             "dph_base": 9.0, "rentable": True, "gpu_ids": [7]},
        ]

    def test_each_card_counted_once(self):
        cards = noise_study.vast_data.physical_gpus(self.nested_offers(), "H100 SXM")
        self.assertEqual(len(cards), 6)

    def test_card_is_free_if_any_offer_with_it_is_rentable(self):
        cards = noise_study.vast_data.physical_gpus(self.nested_offers(), "H100 SXM")
        occupied = {}
        for card in cards:
            occupied[card["gpu_id"]] = card["occupied"]
        self.assertFalse(occupied[4])
        self.assertTrue(occupied[1])
        self.assertTrue(occupied[5])

    def test_machine_units_weight_by_cards(self):
        cards = noise_study.vast_data.physical_gpus(self.nested_offers(), "H100 SXM")
        pairs = sorted(noise_study.units(cards, "machine"))
        self.assertEqual(pairs, [(2.0, 4), (3.0, 2)])
        self.assertEqual(len(noise_study.units(cards, "gpu")), 6)

    def test_host_shares(self):
        cards = noise_study.vast_data.physical_gpus(self.nested_offers(), "H100 SXM")
        shares = noise_study.host_shares(cards)
        self.assertAlmostEqual(shares[0][0], 4 / 6)

    def test_price_shape_floor_share(self):
        shape = noise_study.price_shape([(1.72, 3), (1.72, 1), (2.5, 4)])
        self.assertAlmostEqual(shape["floor_share"], 0.5)
        self.assertEqual(shape["p10"], 1.72)


if __name__ == "__main__":
    unittest.main()

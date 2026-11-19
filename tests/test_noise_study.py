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


class HostTests(unittest.TestCase):
    def test_price_points_group_same_host_same_price(self):
        offers = [
            {"host_id": 1, "dph_base": 1.72, "num_gpus": 1},
            {"host_id": 1, "dph_base": 6.88, "num_gpus": 4},
            {"host_id": 1, "dph_base": 2.50, "num_gpus": 1},
            {"host_id": 2, "dph_base": 1.72, "num_gpus": 1},
        ]
        self.assertEqual(noise_study.price_points(offers), 3)
        self.assertAlmostEqual(noise_study.largest_host_share(offers), 6 / 7)

    def test_price_shape_floor_share(self):
        rentals = [(1.72, 3), (1.72, 1), (2.5, 4)]
        shape = noise_study.price_shape(rentals)
        self.assertAlmostEqual(shape["floor_share"], 0.5)
        self.assertEqual(shape["p10"], 1.72)


if __name__ == "__main__":
    unittest.main()

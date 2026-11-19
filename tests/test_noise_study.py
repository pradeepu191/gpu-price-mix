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


if __name__ == "__main__":
    unittest.main()

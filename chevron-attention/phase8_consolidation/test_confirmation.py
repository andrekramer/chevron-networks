from __future__ import annotations

import unittest

from phase8_consolidation.confirmation import (
    CONFIRMATION_SEEDS,
    ROBUSTNESS_CONDITIONS,
    bootstrap_ci,
    paired_comparison,
)


class ConfirmationTest(unittest.TestCase):
    def test_confirmation_seeds_are_new_and_unique(self) -> None:
        self.assertEqual(len(CONFIRMATION_SEEDS), 20)
        self.assertEqual(len(set(CONFIRMATION_SEEDS)), 20)
        self.assertTrue(all(seed >= 1000 for seed in CONFIRMATION_SEEDS))

    def test_robustness_protocol_contains_boundary_and_pressure_checks(self) -> None:
        names = {item.name for item in ROBUSTNESS_CONDITIONS}
        self.assertTrue({"transient1", "transient4", "transient5"} <= names)
        self.assertTrue({"interleaved", "retained_capacity8", "bank2_interleaved"} <= names)

    def test_bootstrap_is_deterministic(self) -> None:
        first = bootstrap_ci([1.0, 2.0, 3.0], samples=200, seed=9)
        second = bootstrap_ci([1.0, 2.0, 3.0], samples=200, seed=9)
        self.assertEqual(first, second)

    def test_paired_comparison_uses_matching_seeds(self) -> None:
        rows = [
            {"condition": "x", "method": "a", "seed": 1, "metric": 0.8},
            {"condition": "x", "method": "b", "seed": 1, "metric": 0.5},
            {"condition": "x", "method": "a", "seed": 2, "metric": 0.4},
            {"condition": "x", "method": "b", "seed": 2, "metric": 0.5},
        ]
        result = paired_comparison(rows, "x", "a", "b", "metric", 17)
        self.assertEqual(result["n"], 2)
        self.assertAlmostEqual(result["mean"], 0.1)
        self.assertEqual(result["left_wins"], 1)
        self.assertEqual(result["right_wins"], 1)


if __name__ == "__main__":
    unittest.main()


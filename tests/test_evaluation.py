"""Tests for distance and risk evaluation metrics."""

from __future__ import annotations

import unittest

from src.evaluation import critical_recall_false_brake, distance_metrics
from src.risk import ZoneThresholds


class TestEvaluation(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [
            {"ground_truth_distance": 5.0, "estimated_distance": 6.0},
            {"ground_truth_distance": 12.0, "estimated_distance": 10.0},
            {"ground_truth_distance": 35.0, "estimated_distance": 40.0},
        ]

    def test_distance_bins(self) -> None:
        rows = distance_metrics(self.rows)
        self.assertAlmostEqual(rows[0]["MAE_m"], 1.0)
        self.assertEqual(rows[1]["count"], 1)

    def test_critical_recall(self) -> None:
        recall, false_brake = critical_recall_false_brake(self.rows, ZoneThresholds(7.0, 15.0))
        self.assertEqual(recall, 1.0)
        self.assertEqual(false_brake, 0.0)


if __name__ == "__main__":
    unittest.main()

"""Unit tests for corridor inclusion and ignored-region matching."""

from __future__ import annotations

import unittest

from evaluate_corridor import GroundTruth, in_corridor, match_with_ignore_region


class TestCorridorEvaluation(unittest.TestCase):
    def test_primary_corridor_uses_lateral_and_forward_position(self) -> None:
        inside = GroundTruth((10, 10, 30, 60), lateral_m=1.9, forward_m=30.0, occlusion=0)
        outside_lateral = GroundTruth((10, 10, 30, 60), lateral_m=2.1, forward_m=10.0, occlusion=0)
        outside_forward = GroundTruth((10, 10, 30, 60), lateral_m=0.0, forward_m=30.1, occlusion=0)
        self.assertTrue(in_corridor(inside, 2.0))
        self.assertFalse(in_corridor(outside_lateral, 2.0))
        self.assertFalse(in_corridor(outside_forward, 2.0))

    def test_prediction_matching_ignored_ground_truth_is_not_false_positive(self) -> None:
        included = [GroundTruth((0, 0, 20, 40), 0.0, 10.0, 0)]
        ignored = [GroundTruth((30, 0, 50, 40), 4.0, 10.0, 0)]
        predictions = [((30.0, 0.0, 50.0, 40.0), 0.9)]
        tp, fp, fn, pairs = match_with_ignore_region(included, ignored, predictions)
        self.assertEqual((tp, fp, fn, pairs), (0, 0, 1, ()))

    def test_prediction_unmatched_to_included_or_ignored_is_false_positive(self) -> None:
        included = [GroundTruth((0, 0, 20, 40), 0.0, 10.0, 0)]
        ignored = [GroundTruth((30, 0, 50, 40), 4.0, 10.0, 0)]
        predictions = [((80.0, 0.0, 100.0, 40.0), 0.9)]
        self.assertEqual(match_with_ignore_region(included, ignored, predictions)[:3], (0, 1, 1))


if __name__ == "__main__":
    unittest.main()

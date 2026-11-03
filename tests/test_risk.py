"""Unit tests for zone classification and stopping-distance policy."""

from __future__ import annotations

import unittest

from src.risk import ZoneThresholds, classify_zone, get_system_action, stopping_distance, thresholds_from_speed


class TestRisk(unittest.TestCase):
    def test_zone_boundaries(self) -> None:
        limits = ZoneThresholds(critical=5.0, warning=10.0)
        self.assertEqual(classify_zone(4.99, limits).zone, "critical_danger")
        self.assertEqual(classify_zone(5.0, limits).zone, "warning")
        self.assertEqual(classify_zone(10.0, limits).zone, "monitoring")

    def test_stopping_distance_uses_si_units(self) -> None:
        self.assertAlmostEqual(stopping_distance(36.0, 1.0, 6.0), 18.3333333333)

    def test_system_action_is_most_critical(self) -> None:
        self.assertEqual(get_system_action([classify_zone(20), classify_zone(3)]), "EMERGENCY BRAKE")

    def test_speed_thresholds_grow_with_speed(self) -> None:
        self.assertLess(thresholds_from_speed(20).critical, thresholds_from_speed(50).critical)


if __name__ == "__main__":
    unittest.main()

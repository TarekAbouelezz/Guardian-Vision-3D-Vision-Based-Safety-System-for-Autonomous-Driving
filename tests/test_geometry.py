"""Unit tests for camera geometry and Gaussian uncertainty."""

from __future__ import annotations

import unittest

from src.localization.localizer import CameraIntrinsics, Localizer


class TestGeometry(unittest.TestCase):
    def setUp(self) -> None:
        self.localizer = Localizer(CameraIntrinsics(621.0, 621.0, 621.0, 187.5, 1242, 375))

    def test_principal_point_backprojects_on_axis(self) -> None:
        self.assertEqual(self.localizer.backproject([611, 177.5, 631, 197.5], 20.0), (0.0, 0.0, 20.0))

    def test_covariance_contains_squared_standard_deviations(self) -> None:
        gaussian = self.localizer.to_gaussian([611, 177.5, 631, 197.5], 20.0)
        self.assertAlmostEqual(gaussian["covariance"][2][2], 4.0)
        self.assertGreater(gaussian["covariance"][0][0], 0.0)
        self.assertEqual(gaussian["covariance"][0][1], 0.0)


if __name__ == "__main__":
    unittest.main()

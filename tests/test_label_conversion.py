"""Tests for CARLA annotation conversion and sequence splitting helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from convert_labels import convert_line, convert_labels


class TestLabelConversion(unittest.TestCase):
    def test_pedestrian_bbox_is_normalized(self) -> None:
        line = "Pedestrian 0.00 0 3.07 663.84 184.63 668.89 207.87 1.86 0.38 0.38 3.64 1.63 49.87 -3.14"
        self.assertEqual(convert_line(line), "0 0.53652576 0.52333333 0.00406602 0.06197333")

    def test_non_pedestrian_is_skipped(self) -> None:
        self.assertIsNone(convert_line("Car 0.00 0 0.0 1 2 3 4 2 2 4 0 0 10 0"))

    def test_directory_conversion_preserves_empty_frames(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "labels"
            output = root / "yolo"
            source.mkdir()
            (source / "000001.txt").write_text("Car 0 0 0 1 2 3 4 1 1 1 0 0 1 0\n", encoding="utf-8")
            files, boxes = convert_labels(source, output)
            self.assertEqual((files, boxes), (1, 0))
            self.assertEqual((output / "000001.txt").read_text(encoding="utf-8"), "")


if __name__ == "__main__":
    unittest.main()

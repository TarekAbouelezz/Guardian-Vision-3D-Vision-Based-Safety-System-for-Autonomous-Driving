"""Traditional pedestrian-height distance baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from src.depth.base import BaseDepthEstimator


class GeometricDepthEstimator(BaseDepthEstimator):
    """Estimate distance from a known pedestrian height and bbox height."""

    def __init__(self, focal_length_y: float = 621.0, pedestrian_height_m: float = 1.7, output_dir: str = "outputs/depth") -> None:
        self.focal_length_y = focal_length_y
        self.pedestrian_height_m = pedestrian_height_m
        self.output_dir = Path(output_dir)

    def load_model(self) -> None:
        """The geometric baseline has no learned model."""

    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """Return a placeholder map; distances are computed from each bbox."""
        if image is None or image.size == 0:
            raise ValueError("estimate_depth received an empty image")
        return np.ones(image.shape[:2], dtype=np.float32)

    def get_distance(self, depth_map: np.ndarray, bbox: Sequence[float]) -> float:
        """Compute ``Z = fy * pedestrian_height / bbox_height``."""
        height = float(bbox[3]) - float(bbox[1])
        if height <= 0:
            raise ValueError(f"Invalid bbox height: {bbox}")
        return self.focal_length_y * self.pedestrian_height_m / height

    def visualize(self, depth_map: np.ndarray) -> str:
        """Save the constant baseline map for interface compatibility."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / "geometric_depth.png"
        if not cv2.imwrite(str(output_path), np.zeros_like(depth_map, dtype=np.uint8)):
            raise IOError(f"Failed to save depth visualization: {output_path}")
        return str(output_path)

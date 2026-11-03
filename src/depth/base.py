from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np


class BaseDepthEstimator(ABC):
    """Abstract interface for pluggable monocular depth estimators."""

    def estimate(self, image: np.ndarray) -> np.ndarray:
        """Compatibility alias for the canonical dense-depth method."""
        return self.estimate_depth(image)

    @abstractmethod
    def load_model(self) -> None:
        """Load model weights and initialize inference components."""

    @abstractmethod
    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """Estimate a dense depth map from an input RGB/BGR image."""

    @abstractmethod
    def visualize(self, depth_map: np.ndarray) -> str:
        """Save a colorized visualization for a depth map and return its path."""

    def get_distance(self, depth_map: np.ndarray, bbox: Sequence[float]) -> float:
        """Return the median depth in the central quarter of a bounding box."""
        if depth_map.ndim != 2 or depth_map.size == 0:
            raise ValueError("depth_map must be a non-empty 2D array")
        x1, y1, x2, y2 = (int(round(value)) for value in bbox)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        half_w, half_h = max(1, (x2 - x1) // 4), max(1, (y2 - y1) // 4)
        crop = depth_map[
            max(0, cy - half_h):min(depth_map.shape[0], cy + half_h + 1),
            max(0, cx - half_w):min(depth_map.shape[1], cx + half_w + 1),
        ]
        values = crop[np.isfinite(crop)]
        if values.size == 0:
            raise ValueError(f"No finite depth values in bbox {bbox}")
        return float(np.median(values))

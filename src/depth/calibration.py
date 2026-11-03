"""Training-split calibration for relative monocular depth predictions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class LinearDepthCalibration:
    """Affine mapping fitted only from training observations."""

    scale: float
    offset: float

    @classmethod
    def fit(cls, relative_depth: np.ndarray, metric_distance: np.ndarray) -> "LinearDepthCalibration":
        """Fit ``metric_distance = scale * relative_depth + offset``."""
        x = np.asarray(relative_depth, dtype=np.float64).reshape(-1)
        y = np.asarray(metric_distance, dtype=np.float64).reshape(-1)
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() < 2:
            raise ValueError("At least two finite training observations are required")
        design = np.column_stack((x[valid], np.ones(valid.sum())))
        scale, offset = np.linalg.lstsq(design, y[valid], rcond=None)[0]
        if scale <= 0:
            raise ValueError("Fitted depth mapping must have a positive scale")
        return cls(float(scale), float(offset))

    def apply(self, relative_depth: float | np.ndarray) -> float | np.ndarray:
        """Map relative depth to metric distance."""
        return self.scale * relative_depth + self.offset

    def save(self, path: Path) -> None:
        """Persist calibration parameters."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"scale": self.scale, "offset": self.offset}, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "LinearDepthCalibration":
        """Load calibration parameters from JSON."""
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(float(raw["scale"]), float(raw["offset"]))

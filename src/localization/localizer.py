"""Back-projection and uncertainty estimation for detected pedestrians."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from src.localization.models import PedestrianLocalization

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole camera intrinsics in pixel units."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


class Localizer:
    """Convert 2D detections and metric depth into 3D Gaussian objects."""

    PERSON_CLASS_NAME = "person"

    def __init__(self, intrinsics: CameraIntrinsics, output_dir: str = "outputs/localization") -> None:
        self.K = intrinsics
        self.output_dir = Path(output_dir)

    def backproject(self, bbox: Sequence[float], depth: float) -> tuple[float, float, float]:
        """Back-project the image-box centre into camera coordinates."""
        if depth <= 0:
            raise ValueError(f"Depth must be positive, got {depth}")
        u = (float(bbox[0]) + float(bbox[2])) / 2.0
        v = (float(bbox[1]) + float(bbox[3])) / 2.0
        z = float(depth)
        return ((u - self.K.cx) * z / self.K.fx, (v - self.K.cy) * z / self.K.fy, z)

    def to_gaussian(self, bbox: Sequence[float], depth: float) -> dict[str, list[list[float]] | list[float]]:
        """Build a diagonal 3D Gaussian with variance, not standard deviation, entries."""
        mean = self.backproject(bbox, depth)
        width_px = float(bbox[2]) - float(bbox[0])
        height_px = float(bbox[3]) - float(bbox[1])
        if width_px <= 0 or height_px <= 0:
            raise ValueError(f"Invalid bounding box: {bbox}")
        width_m = width_px * depth / self.K.fx
        height_m = height_px * depth / self.K.fy
        sigma_x = max(width_m / 2.0, 0.01)
        sigma_y = max(height_m / 2.0, 0.01)
        sigma_z = max(depth * 0.10, 0.05)
        covariance = [
            [sigma_x**2, 0.0, 0.0],
            [0.0, sigma_y**2, 0.0],
            [0.0, 0.0, sigma_z**2],
        ]
        return {"mean": list(mean), "covariance": covariance}

    def localize(self, detections: Sequence[Mapping[str, Any]], depth_map: np.ndarray) -> list[PedestrianLocalization]:
        """Create ordered localization records using central-region depth medians."""
        if depth_map is None or depth_map.ndim != 2 or depth_map.size == 0:
            raise ValueError("localize requires a non-empty 2D depth map")
        localized: list[PedestrianLocalization] = []
        for detection in detections:
            if str(detection.get("class_name", "")).lower() != self.PERSON_CLASS_NAME:
                continue
            bbox = self._parse_bbox(detection.get("bbox"))
            if bbox is None:
                LOGGER.warning("Skipping detection with invalid bbox: %s", detection)
                continue
            values = self._central_depth_values(depth_map, bbox)
            if values.size == 0:
                LOGGER.warning("Skipping pedestrian with no finite depth: %s", detection)
                continue
            x1, y1, x2, y2 = bbox
            localized.append(PedestrianLocalization(
                id=len(localized) + 1,
                class_name=str(detection.get("class_name", self.PERSON_CLASS_NAME)),
                confidence=float(detection.get("confidence", 0.0)),
                bbox=[x1, y1, x2, y2],
                ground_point=((x1 + x2) // 2, y2),
                relative_depth=float(np.median(values)),
            ))
        localized.sort(key=lambda item: item.relative_depth)
        for index, item in enumerate(localized, start=1):
            item.id = index
        return localized

    def visualize(self, image: np.ndarray, localizations: Sequence[PedestrianLocalization], output_name: str | None = None) -> str:
        """Save simple localization overlays."""
        if image is None or image.size == 0:
            raise ValueError("visualize received an empty image")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / (output_name or f"localization_{datetime.now():%Y%m%d_%H%M%S}.png")
        annotated = image.copy()
        for localization in localizations:
            x1, y1, x2, y2 = localization.bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 180, 255), 2)
            cv2.putText(annotated, f"Person #{localization.id} {localization.relative_depth:.2f}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 180, 255), 2, cv2.LINE_AA)
        if not cv2.imwrite(str(output_path), annotated):
            raise IOError(f"Failed to save localization visualization to {output_path}")
        return str(output_path)

    @staticmethod
    def _parse_bbox(bbox: Any) -> tuple[int, int, int, int] | None:
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return None
        try:
            x1, y1, x2, y2 = (int(round(float(value))) for value in bbox)
        except (TypeError, ValueError):
            return None
        return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None

    @staticmethod
    def _central_depth_values(depth_map: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
        """Extract the central 50% box region and remove invalid values."""
        x1, y1, x2, y2 = bbox
        height, width = depth_map.shape
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            return np.array([], dtype=np.float32)
        margin_x, margin_y = max(1, (x2 - x1) // 4), max(1, (y2 - y1) // 4)
        values = depth_map[y1 + margin_y:y2 - margin_y + 1, x1 + margin_x:x2 - margin_x + 1]
        return values[np.isfinite(values) & (values > 0)]

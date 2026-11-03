from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from src.depth.base import BaseDepthEstimator
from src.depth.calibration import LinearDepthCalibration


LOGGER = logging.getLogger(__name__)


class DepthAnythingEstimator(BaseDepthEstimator):
    """Depth estimator backed by Depth Anything V2 Hugging Face weights."""

    DEFAULT_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"

    def __init__(
        self,
        model_path: str = "",
        device: str = "cpu",
        output_dir: str = "outputs/depth",
        calibration_path: str = "",
    ) -> None:
        self.model_path = model_path.strip() if model_path else self.DEFAULT_MODEL_ID
        self.output_dir = Path(output_dir)
        self.device = self._resolve_device(device)
        self.calibration = LinearDepthCalibration.load(Path(calibration_path)) if calibration_path else None
        self.image_processor: AutoImageProcessor | None = None
        self.model: AutoModelForDepthEstimation | None = None
        self.load_model()

    def _resolve_device(self, requested_device: str) -> str:
        normalized = requested_device.strip().lower()
        if normalized == "cuda" and not torch.cuda.is_available():
            LOGGER.warning("CUDA requested but unavailable. Falling back to CPU.")
            return "cpu"
        return normalized

    def load_model(self) -> None:
        try:
            self.image_processor = AutoImageProcessor.from_pretrained(self.model_path)
            self.model = AutoModelForDepthEstimation.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            LOGGER.info("Loaded depth model %s on %s", self.model_path, self.device)
        except Exception as exc:
            LOGGER.exception("Failed to load depth model: %s", self.model_path)
            raise RuntimeError(f"Unable to load depth model: {self.model_path}") from exc

    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            raise ValueError("estimate_depth received an empty image")
        if self.image_processor is None or self.model is None:
            raise RuntimeError("Depth model has not been loaded")

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        inputs = self.image_processor(images=rgb_image, return_tensors="pt")
        inputs = {key: tensor.to(self.device) for key, tensor in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth

        resized = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=rgb_image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        return resized.detach().cpu().numpy().astype(np.float32)

    def estimate(self, image: np.ndarray) -> np.ndarray:
        return self.estimate_depth(image)

    def get_distance(self, depth_map: np.ndarray, bbox: Sequence[float]) -> float:
        x1, y1, x2, y2 = [int(v) for v in bbox]

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        half_w = max(1, (x2 - x1) // 4)
        half_h = max(1, (y2 - y1) // 4)

        crop = depth_map[
            max(0, cy - half_h): min(depth_map.shape[0], cy + half_h),
            max(0, cx - half_w): min(depth_map.shape[1], cx + half_w),
        ]

        if crop.size == 0:
            return float(depth_map[min(max(cy, 0), depth_map.shape[0] - 1), min(max(cx, 0), depth_map.shape[1] - 1)])
        relative = float(np.median(crop))
        return float(self.calibration.apply(relative)) if self.calibration else relative

    def visualize(self, depth_map: np.ndarray) -> str:
        if depth_map is None or depth_map.size == 0:
            raise ValueError("visualize received an empty depth map")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        depth_min = float(np.min(depth_map))
        depth_max = float(np.max(depth_map))

        if depth_max - depth_min < 1e-8:
            normalized = np.zeros_like(depth_map, dtype=np.uint8)
        else:
            scaled = (depth_map - depth_min) / (depth_max - depth_min)
            normalized = np.clip(scaled * 255.0, 0, 255).astype(np.uint8)

        colorized = cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"depth_{timestamp}.png"

        if not cv2.imwrite(str(output_path), colorized):
            raise IOError(f"Failed to save depth visualization: {output_path}")

        LOGGER.info("Saved depth visualization to %s", output_path)
        return str(output_path)

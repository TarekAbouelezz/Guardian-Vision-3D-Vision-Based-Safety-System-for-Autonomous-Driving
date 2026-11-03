"""Calibrated transformer depth backends used by MiDaS and Metric3D options."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from src.depth.base import BaseDepthEstimator
from src.depth.calibration import LinearDepthCalibration

LOGGER = logging.getLogger(__name__)


class TransformerRelativeEstimator(BaseDepthEstimator):
    """Run a relative-depth transformer and apply a training-fitted mapping."""

    def __init__(self, model_path: str, device: str, output_dir: str, calibration_path: str = "") -> None:
        self.model_path = model_path
        self.device = "cuda" if device.lower() == "cuda" and torch.cuda.is_available() else "cpu"
        self.output_dir = Path(output_dir)
        self.calibration = LinearDepthCalibration.load(Path(calibration_path)) if calibration_path else None
        self.processor = AutoImageProcessor.from_pretrained(model_path)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_path).to(self.device).eval()

    def load_model(self) -> None:
        """Model loading occurs during construction to surface failures early."""

    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """Return a dense relative or calibrated metric depth map."""
        if image is None or image.size == 0:
            raise ValueError("estimate_depth received an empty image")
        inputs = self.processor(images=cv2.cvtColor(image, cv2.COLOR_BGR2RGB), return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            predicted = self.model(**inputs).predicted_depth
        depth = torch.nn.functional.interpolate(predicted.unsqueeze(1), size=image.shape[:2], mode="bicubic", align_corners=False).squeeze().cpu().numpy()
        if self.calibration:
            depth = self.calibration.apply(depth)
        return depth.astype(np.float32)

    def visualize(self, depth_map: np.ndarray) -> str:
        """Persist a colorized depth map."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        output_path = self.output_dir / f"{self.__class__.__name__}.png"
        if not cv2.imwrite(str(output_path), cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)):
            raise IOError(f"Failed to save depth visualization: {output_path}")
        return str(output_path)

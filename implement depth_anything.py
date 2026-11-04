import torch
import numpy as np
import cv2
from .base import BaseDepthEstimator


class DepthAnythingEstimator(BaseDepthEstimator):
    """
    Monocular depth estimator using Depth Anything V2.
    Returns a relative depth map; scale is calibrated using
    known ground-truth distances from the CARLA dataset.
    """

    def __init__(self, model_size: str = "small", device: str = "cpu"):
        self.device = device
        self.model_size = model_size
        self.model = self._load_model()
        self.scale_factor = 1.0   # calibrated in calibrate()

    def _load_model(self):
        from transformers import pipeline
        model_id = {
            "small":  "depth-anything/Depth-Anything-V2-Small-hf",
            "base":   "depth-anything/Depth-Anything-V2-Base-hf",
            "large":  "depth-anything/Depth-Anything-V2-Large-hf",
        }[self.model_size]

        pipe = pipeline(
            task="depth-estimation",
            model=model_id,
            device=0 if self.device == "cuda" else -1
        )
        return pipe

    def estimate(self, image: np.ndarray) -> np.ndarray:
        """
        Run depth estimation on a BGR numpy image.
        Returns a float32 depth map in the same H×W as input.
        Values are relative (0–1 range) unless calibrate() was called.
        """
        from PIL import Image as PILImage
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(rgb)

        result = self.model(pil_img)
        depth = np.array(result["depth"], dtype=np.float32)

        # Resize to match original image size
        depth = cv2.resize(depth, (image.shape[1], image.shape[0]))

        # Apply scale factor if calibrated
        depth = depth * self.scale_factor
        return depth

    def get_distance(self, depth_map: np.ndarray, bbox: list) -> float:
        """
        Extract the estimated distance for a detected object
        given its 2D bounding box [x1, y1, x2, y2].
        Uses median over the central 50% of the box to avoid edge noise.
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]

        # Shrink to central 50% of box to avoid background contamination
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        half_w = (x2 - x1) // 4
        half_h = (y2 - y1) // 4

        crop = depth_map[
            max(0, cy - half_h): cy + half_h,
            max(0, cx - half_w): cx + half_w
        ]

        if crop.size == 0:
            return float(depth_map[cy, cx])

        return float(np.median(crop))

    def calibrate(self, depth_map: np.ndarray,
                  bbox: list, true_distance: float):
        """
        Calibrate the scale factor using one known ground-truth distance.
        Call this once on a CARLA frame where you know the real distance.
        E.g.: calibrate(depth_map, bbox_of_person, true_distance=3.0)
        """
        raw = self.get_distance(depth_map, bbox) / self.scale_factor
        if raw > 0:
            self.scale_factor = true_distance / raw
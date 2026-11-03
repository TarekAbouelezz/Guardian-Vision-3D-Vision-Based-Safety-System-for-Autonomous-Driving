from __future__ import annotations

from src.depth.relative_transformer import TransformerRelativeEstimator


class MiDaSEstimator(TransformerRelativeEstimator):
    """MiDaS/DPT relative estimator with optional training calibration."""

    def __init__(self, model_path: str = "Intel/dpt-hybrid-midas", device: str = "cpu", output_dir: str = "outputs/depth", calibration_path: str = "") -> None:
        super().__init__(model_path, device, output_dir, calibration_path)


class ZoeDepthEstimator(MiDaSEstimator):
    """Compatibility alias using the same calibrated relative-depth contract."""

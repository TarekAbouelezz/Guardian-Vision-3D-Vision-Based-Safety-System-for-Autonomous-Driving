from __future__ import annotations

from src.depth.relative_transformer import TransformerRelativeEstimator


class Metric3DEstimator(TransformerRelativeEstimator):
    """Metric-depth backend with the same training-only calibration contract."""

    def __init__(self, model_path: str = "Intel/dpt-large", device: str = "cpu", output_dir: str = "outputs/depth", calibration_path: str = "") -> None:
        super().__init__(model_path, device, output_dir, calibration_path)

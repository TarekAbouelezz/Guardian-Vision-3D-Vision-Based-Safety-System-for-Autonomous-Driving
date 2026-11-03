from __future__ import annotations

from dataclasses import dataclass

from src.depth.base import BaseDepthEstimator
from src.depth.depth_anything import DepthAnythingEstimator
from src.depth.geometric import GeometricDepthEstimator
from src.depth.metric3d import Metric3DEstimator
from src.depth.midas import MiDaSEstimator, ZoeDepthEstimator


@dataclass(frozen=True)
class DepthConfig:
    """Depth runtime configuration used by DepthFactory."""

    model: str = "depth_anything"
    model_path: str = ""
    device: str = "cpu"
    output_dir: str = "outputs/depth"
    calibration_path: str = ""


class _NotImplementedDepthEstimator(BaseDepthEstimator):
    """Placeholder estimator used for planned but unimplemented models."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def load_model(self) -> None:
        raise NotImplementedError(f"{self.model_name} is not implemented yet.")

    def estimate_depth(self, image) -> object:
        raise NotImplementedError(f"{self.model_name} is not implemented yet.")

    def visualize(self, depth_map) -> str:
        raise NotImplementedError(f"{self.model_name} is not implemented yet.")


class DepthFactory:
    """Factory for constructing depth estimator implementations by name."""

    @staticmethod
    def create(config: DepthConfig) -> BaseDepthEstimator:
        model_name = config.model.strip().lower()

        if model_name in {"depth_anything", "depthanything"}:
            return DepthAnythingEstimator(
                model_path=config.model_path,
                device=config.device,
                output_dir=config.output_dir,
                calibration_path=config.calibration_path,
            )

        if model_name in {"geometric", "geometry"}:
            return GeometricDepthEstimator(output_dir=config.output_dir)

        if model_name in {"metric3d", "metric_3d"}:
            return Metric3DEstimator(config.model_path or "Intel/dpt-large", config.device, config.output_dir, config.calibration_path)

        if model_name == "midas":
            return MiDaSEstimator(config.model_path or "Intel/dpt-hybrid-midas", config.device, config.output_dir, config.calibration_path)

        if model_name in {"zoedepth", "zoe_depth"}:
            return ZoeDepthEstimator(config.model_path or "Intel/dpt-hybrid-midas", config.device, config.output_dir, config.calibration_path)

        raise ValueError(f"Unsupported depth model: {config.model}")

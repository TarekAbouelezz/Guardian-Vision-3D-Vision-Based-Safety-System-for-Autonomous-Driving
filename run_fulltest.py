"""Run the complete image directory through the Guardian Vision pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

from src.config.loader import load_config
from src.depth.factory import DepthConfig, DepthFactory
from src.detection.factory import DetectionFactory, DetectorConfig
from src.localization.localizer import CameraIntrinsics, Localizer
from src.main import DEFAULT_INTRINSICS, run_image_pipeline
from src.risk import ZoneThresholds


def main() -> None:
    """Process every source image once with one loaded detector and estimator."""
    root = Path(__file__).resolve().parent
    config = load_config(root / "src/config/config.yaml")
    output_dir = root / "outputs/fulltest_6566"
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(root / "outputs/fulltest_6566.log", mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    detector = DetectionFactory.create(DetectorConfig(
        model=config.detector.model,
        model_path=config.detector.model_path,
        confidence_threshold=config.confidence_threshold,
        device=config.device,
        output_images_dir=str(output_dir),
        output_videos_dir=config.paths.output_videos,
    ))
    depth_estimator = DepthFactory.create(DepthConfig(
        model="geometric",
        model_path="",
        device=config.device,
        output_dir=str(root / "outputs/depth"),
        calibration_path="",
    ))
    localizer = Localizer(intrinsics=DEFAULT_INTRINSICS)
    thresholds = ZoneThresholds(config.risk.critical_distance_m, config.risk.warning_distance_m)
    images = sorted(
        path for path in (root / "dataset/images").iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    logging.info("Starting full test: %d images", len(images))
    for index, image_path in enumerate(images, start=1):
        run_image_pipeline(image_path, detector, depth_estimator, localizer, output_dir, thresholds)
        if index % 100 == 0 or index == len(images):
            logging.info("Completed %d/%d images", index, len(images))
    logging.info("Full test complete: %d images", len(images))


if __name__ == "__main__":
    main()

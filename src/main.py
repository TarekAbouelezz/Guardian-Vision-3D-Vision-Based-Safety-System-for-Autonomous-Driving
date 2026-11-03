from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from src.config.loader import load_config
from src.detection.base import Detection
from src.detection.factory import DetectionFactory, DetectorConfig
from src.depth.factory import DepthFactory, DepthConfig
from src.localization.localizer import Localizer, CameraIntrinsics
from src.risk import ZoneResult, ZoneThresholds, classify_zone, get_system_action


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Guardian Vision - Modular Research Framework"
    )
    parser.add_argument("--config",  type=str, default="src/config/config.yaml")
    parser.add_argument(
        "--image",
        type=str,
        nargs="+",
        default=None,
        help="One or more image paths to process.",
    )
    parser.add_argument("--video",   type=str, default=None)
    parser.add_argument("--model",   type=str, default=None)
    return parser.parse_args()


# ── Camera intrinsics (CARLA default, 1242×375, fov=90) ───────────────────────
# Override these with values from your calib/*.txt files if needed.
DEFAULT_INTRINSICS = CameraIntrinsics(
    fx=621.0, fy=621.0,
    cx=621.0, cy=187.5,
    width=1242, height=375,
)


# ── Visualization ─────────────────────────────────────────────────────────────

ZONE_COLORS = {
    "monitoring":     (0,   255,   0),   # green
    "warning":        (0,   165, 255),   # orange
    "critical_danger":(0,     0, 255),   # red
}

ZONE_LABELS = {
    "monitoring":      "MONITORING  — ALARM",
    "warning":         "WARNING     — REDUCE SPEED",
    "critical_danger": "DANGER      — EMERGENCY BRAKE",
}


def draw_zone_detections(
    image: np.ndarray,
    detections: Sequence[Detection],
    zone_results: list[ZoneResult],
) -> np.ndarray:
    """
    Draw bounding boxes colored by zone, with distance and action label.
    Also draws a semi-transparent zone legend in the top-left corner.
    """
    vis = image.copy()
    h, w = vis.shape[:2]

    for det, zone in zip(detections, zone_results):
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        color = ZONE_COLORS[zone.zone]

        # Bounding box
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness=2)

        # Label: zone + distance
        label = f"{zone.zone.replace('_', ' ').upper()}  {zone.distance:.1f}m"
        label_y = max(y1 - 8, 16)
        cv2.putText(
            vis, label, (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA
        )

        # Draw Gaussian ellipse (uncertainty in XY plane projected to image)
        cx_box = (x1 + x2) // 2
        cy_box = (y1 + y2) // 2
        axes = (max(4, (x2 - x1) // 2), max(4, (y2 - y1) // 2))
        cv2.ellipse(vis, (cx_box, cy_box), axes, 0, 0, 360, color, 1)

    # ── System action banner ──────────────────────────────────────────────────
    if zone_results:
        system_action = get_system_action(zone_results)
        most_critical = min(zone_results, key=lambda z: z.distance)
        banner_color = ZONE_COLORS[most_critical.zone]

        overlay = vis.copy()
        cv2.rectangle(overlay, (0, h - 44), (w, h), banner_color, -1)
        cv2.addWeighted(overlay, 0.5, vis, 0.5, 0, vis)
        cv2.putText(
            vis,
            f"SYSTEM ACTION: {system_action}",
            (12, h - 14),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
        )

    # ── Zone legend ───────────────────────────────────────────────────────────
    for i, (zone_key, label_text) in enumerate(ZONE_LABELS.items()):
        y_pos = 20 + i * 22
        cv2.rectangle(vis, (8, y_pos - 12), (18, y_pos), ZONE_COLORS[zone_key], -1)
        cv2.putText(
            vis, label_text, (24, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA
        )

    return vis


# ── Console summary ───────────────────────────────────────────────────────────

def print_detection_summary(
    detections: Sequence[Detection],
    zone_results: list[ZoneResult],
) -> None:
    print("\n" + "=" * 55)
    print("  GUARDIAN VISION — Detection Summary")
    print("=" * 55)

    if not detections:
        print("  No persons detected.")
        print("=" * 55)
        return

    for i, (det, zone) in enumerate(zip(detections, zone_results), 1):
        x1, y1, x2, y2 = det["bbox"]
        print(f"\n  Person {i}")
        print(f"    Confidence : {det['confidence']:.2f}")
        print(f"    BBox       : ({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f})")
        print(f"    Distance   : {zone.distance:.2f} m")
        print(f"    Zone       : {zone.zone.replace('_', ' ').upper()}")
        print(f"    Action     : {zone.action}")

    system_action = get_system_action(zone_results)
    print(f"\n  >>> SYSTEM ACTION: {system_action}")
    print("=" * 55 + "\n")


# ── Image pipeline ────────────────────────────────────────────────────────────

def run_image_pipeline(
    image_path: Path,
    detector,
    depth_estimator,
    localizer: Localizer,
    output_dir: Path,
    risk_thresholds: ZoneThresholds | None = None,
) -> None:
    logger = logging.getLogger("guardian_vision.pipeline")

    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Failed to load image: {image_path}")

    logger.info("Running detection on %s", image_path.name)
    detections = detector.detect_image(str(image_path))

    logger.info("Running depth estimation")
    depth_map = depth_estimator.estimate(image)

    # Build zone result for each detection
    zone_results: list[ZoneResult] = []
    for det in detections:
        distance = depth_estimator.get_distance(depth_map, det["bbox"])
        position_3d = localizer.backproject(det["bbox"], distance)
        gaussian    = localizer.to_gaussian(det["bbox"], distance)
        zone        = classify_zone(distance, risk_thresholds)

        logger.info(
            "Person at %.2fm → zone=%s | 3D=(%.1f, %.1f, %.1f)",
            distance, zone.zone, *position_3d
        )
        zone_results.append(zone)

    # Draw and save
    vis = draw_zone_detections(image, detections, zone_results)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{image_path.stem}_guardian.jpg"
    cv2.imwrite(str(out_path), vis)

    print_detection_summary(detections, zone_results)
    print(f"Saved annotated image → {out_path}")


# ── Video pipeline ────────────────────────────────────────────────────────────

def run_video_pipeline(
    video_path: str,
    detector,
    depth_estimator,
    localizer: Localizer,
    output_dir: Path,
    risk_thresholds: ZoneThresholds | None = None,
) -> None:
    logger = logging.getLogger("guardian_vision.video")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 20.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{Path(video_path).stem}_guardian.mp4"
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (width, height)
    )

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections  = detector.detect_image_array(frame)
        depth_map   = depth_estimator.estimate(frame)

        zone_results = []
        for det in detections:
            distance = depth_estimator.get_distance(depth_map, det["bbox"])
            zone     = classify_zone(distance, risk_thresholds)
            zone_results.append(zone)

        vis = draw_zone_detections(frame, detections, zone_results)
        writer.write(vis)
        frame_idx += 1

        if frame_idx % 30 == 0:
            logger.info("Processed %d frames", frame_idx)

    cap.release()
    writer.release()
    logger.info("Saved video → %s", out_path)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    configure_logging()
    args   = parse_args()
    logger = logging.getLogger("guardian_vision.main")

    config = load_config(args.config)

    image_paths = [Path(path) for path in (args.image or [config.paths.image])]
    model_path_value = args.model or config.detector.model_path

    # ── Detector ──────────────────────────────────────────────────────────────
    detector_config = DetectorConfig(
        model=config.detector.model,
        model_path=model_path_value,
        confidence_threshold=config.confidence_threshold,
        device=config.device,
        output_images_dir=config.paths.output_images,
        output_videos_dir=config.paths.output_videos,
    )
    detector = DetectionFactory.create(detector_config)

    # ── Depth estimator ───────────────────────────────────────────────────────
    depth_config = DepthConfig(
        model=config.depth.model,           # "depth_anything" | "midas" | "metric3d"
        model_path=config.depth.model_path,
        device=config.device,
        calibration_path=config.depth.calibration_path,
    )
    depth_estimator = DepthFactory.create(depth_config)
    risk_thresholds = ZoneThresholds(
        critical=config.risk.critical_distance_m,
        warning=config.risk.warning_distance_m,
    )

    # ── Localizer (camera intrinsics) ─────────────────────────────────────────
    localizer = Localizer(intrinsics=DEFAULT_INTRINSICS)

    # ── Run pipelines ─────────────────────────────────────────────────────────
    output_images_dir = Path(config.paths.output_images)

    for image_path in image_paths:
        if not image_path.exists():
            raise FileNotFoundError(f"Input image not found: {image_path}")
        run_image_pipeline(
            image_path, detector, depth_estimator,
            localizer, output_images_dir, risk_thresholds
        )

    if args.video:
        run_video_pipeline(
            args.video, detector, depth_estimator,
            localizer, output_images_dir.parent / "videos", risk_thresholds
        )


if __name__ == "__main__":
    main()

"""Audit YOLO detection, depth, and safety metrics on the full dataset.

The audit writes to ``outputs/audit`` by default and never overwrites the
legacy ``outputs/results/full`` report.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
from ultralytics import YOLO

from finetune import DISTANCE_BINS, iou, parse_pedestrians
from src.risk import ZoneThresholds, classify_zone

LOGGER = logging.getLogger(__name__)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
HEIGHT_BINS = (("<25", 0.0, 25.0), ("25-40", 25.0, 40.0), (">=40", 40.0, float("inf")))
OCCLUSION_LEVELS = ("0", "1", "2", "3+")


@dataclass(frozen=True)
class FrameData:
    """Ground truth and predictions for one image."""

    image_path: Path
    ground_truth: list[tuple[float, float, float, float, float, int, str]]
    predictions: list[tuple[list[float], float]]


def _kitti_rows(label_path: Path) -> list[tuple[str, list[float]]]:
    """Read all valid KITTI rows, retaining the object type."""
    rows = []
    if not label_path.exists():
        return rows
    for line in label_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 15:
            continue
        try:
            values = [float(value) for value in fields[1:]]
        except ValueError:
            continue
        rows.append((fields[0], values))
    return rows


def _label_statistics(labels_dir: Path) -> dict[str, object]:
    """Summarize object types, invalid boxes, duplicates, and distributions."""
    types: Counter[str] = Counter()
    duplicate_frames = 0
    duplicate_pairs = 0
    zero_boxes = 0
    near_zero_boxes = 0
    distances: list[float] = []
    occlusions: Counter[str] = Counter()
    heights: list[float] = []
    pedestrian_count = 0
    for label_path in sorted(labels_dir.glob("*.txt")):
        rows = _kitti_rows(label_path)
        boxes: list[list[float]] = []
        for object_type, values in rows:
            types[object_type] += 1
            x1, y1, x2, y2 = values[3:7]
            width, height = x2 - x1, y2 - y1
            if width <= 0 or height <= 0:
                zero_boxes += 1
            if width < 1 or height < 1:
                near_zero_boxes += 1
            boxes.append([x1, y1, x2, y2])
            if object_type.lower() == "pedestrian":
                pedestrian_count += 1
                distances.append(values[12])
                occlusion = int(values[1])
                occlusions[str(occlusion) if occlusion <= 2 else "3+"] += 1
                heights.append(height)
        frame_duplicates = sum(
            iou(left, right) > 0.9
            for index, left in enumerate(boxes)
            for right in boxes[index + 1:]
        )
        if frame_duplicates:
            duplicate_frames += 1
            duplicate_pairs += frame_duplicates

    def distribution(values: Sequence[float]) -> dict[str, float]:
        if not values:
            return {"count": 0}
        array = np.asarray(values, dtype=float)
        return {
            "count": int(array.size),
            "min": float(np.min(array)),
            "p25": float(np.percentile(array, 25)),
            "median": float(np.median(array)),
            "mean": float(np.mean(array)),
            "p75": float(np.percentile(array, 75)),
            "max": float(np.max(array)),
        }

    return {
        "object_counts_by_type": dict(types),
        "pedestrian_count": pedestrian_count,
        "zero_or_negative_size_boxes": zero_boxes,
        "near_zero_boxes_lt_1px": near_zero_boxes,
        "duplicate_frames_iou_gt_0.9": duplicate_frames,
        "duplicate_pairs_iou_gt_0.9": duplicate_pairs,
        "pedestrian_distance_m": distribution(distances),
        "pedestrian_box_height_px": distribution(heights),
        "pedestrian_occlusion": dict(occlusions),
        "walker_provenance": {
            "controller_ai_walker_in_collector": True,
            "label_level_provenance_available": False,
            "evidence": "collect_data.py spawns walker.pedestrian.* actors and controller.ai.walker controllers; labels contain projected boxes but no actor provenance field.",
        },
    }


def _ground_truth(label_path: Path) -> list[tuple[float, float, float, float, float, int, str]]:
    """Read pedestrian boxes as box, distance, occlusion, height."""
    result = []
    for object_type, values in _kitti_rows(label_path):
        if object_type.lower() != "pedestrian":
            continue
        result.append((values[3], values[4], values[5], values[6], values[12], int(values[1]), object_type))
    return result


def _criterion(predicted: Sequence[float], truth: Sequence[float], mode: str) -> float:
    """Return a match score, or -1 when the centre criterion is not met."""
    if mode == "iou":
        return iou(predicted, truth)
    centre_x = (predicted[0] + predicted[2]) / 2.0
    centre_y = (predicted[1] + predicted[3]) / 2.0
    return 1.0 if truth[0] <= centre_x <= truth[2] and truth[1] <= centre_y <= truth[3] else -1.0


def _match(
    ground_truth: Sequence[tuple[float, float, float, float, float, int, str]],
    predictions: Sequence[tuple[list[float], float]],
    threshold: float,
    mode: str,
) -> tuple[dict[int, int], dict[int, int]]:
    """Perform confidence-ordered one-to-one matching against all GT boxes."""
    matched_predictions: dict[int, int] = {}
    matched_truth: dict[int, int] = {}
    for prediction_index in sorted(range(len(predictions)), key=lambda index: -predictions[index][1]):
        candidates = [
            (_criterion(predictions[prediction_index][0], truth[:4], mode), truth_index)
            for truth_index, truth in enumerate(ground_truth)
            if truth_index not in matched_truth
        ]
        if not candidates:
            continue
        score, truth_index = max(candidates)
        if score >= threshold:
            matched_predictions[prediction_index] = truth_index
            matched_truth[truth_index] = prediction_index
    return matched_predictions, matched_truth


def _stratum(truth: tuple[float, float, float, float, float, int, str], dimension: str) -> str:
    """Return a requested reporting stratum."""
    height = truth[3] - truth[1]
    if dimension == "distance":
        return next(name for low, high, name in DISTANCE_BINS if low <= truth[4] < high)
    if dimension == "occlusion":
        return str(truth[5]) if truth[5] <= 2 else "3+"
    return next(name for name, low, high in HEIGHT_BINS if low <= height < high)


def _metrics(
    frames: Sequence[FrameData],
    threshold: float,
    mode: str,
    subset: str | None = None,
    dimension: str | None = None,
) -> dict[str, float | int | str]:
    """Calculate precision, recall, and distance MAE for a subset."""
    tp = fp = fn = 0
    errors: list[float] = []
    for frame in frames:
        included = []
        for index, truth in enumerate(frame.ground_truth):
            height = truth[3] - truth[1]
            included_by_difficulty = dimension is not None or (
                subset is None
                or subset == "easy" and height >= 40 and truth[5] == 0
                or subset == "moderate" and height >= 25 and truth[5] <= 1
                or subset == "hard" and height >= 25 and truth[5] <= 2
            )
            included_by_stratum = dimension is None or _stratum(truth, dimension) == subset
            if included_by_difficulty and included_by_stratum:
                included.append(index)
        all_matches, all_truth_matches = _match(frame.ground_truth, frame.predictions, threshold, mode)
        included_set = set(included)
        included_matches = {
            truth_index: prediction_index
            for truth_index, prediction_index in all_truth_matches.items()
            if truth_index in included_set
        }
        ignored_prediction_indices = {
            prediction_index
            for truth_index, prediction_index in all_truth_matches.items()
            if truth_index not in included_set
        }
        tp += len(included_matches)
        fn += len(included) - len(included_matches)
        fp += len(frame.predictions) - len(ignored_prediction_indices) - len(included_matches)
        for truth_index, prediction_index in included_matches.items():
            predicted_box = frame.predictions[prediction_index][0]
            height = predicted_box[3] - predicted_box[1]
            if height > 0:
                errors.append(abs(621.0 * 1.7 / height - frame.ground_truth[truth_index][4]))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "subset": subset or "unfiltered",
        "threshold": threshold,
        "matching": mode,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "distance_mae_m": float(np.mean(errors)) if errors else float("nan"),
    }


def _critical_metrics(frames: Sequence[FrameData], threshold: float = 0.5) -> dict[str, float | int]:
    """Use every critical GT pedestrian as the recall denominator."""
    critical_truth = noncritical_truth = matched_critical = predicted_critical_noncritical = 0
    frames_without_critical = frames_with_false_brake = 0
    for frame in frames:
        matches, truth_matches = _match(frame.ground_truth, frame.predictions, threshold, "iou")
        frame_has_critical = any(classify_zone(truth[4]).zone == "critical_danger" for truth in frame.ground_truth)
        frame_has_predicted_critical = False
        for prediction_index, (box, _) in enumerate(frame.predictions):
            height = box[3] - box[1]
            if height <= 0:
                continue
            distance = 621.0 * 1.7 / height
            if classify_zone(distance).zone == "critical_danger":
                frame_has_predicted_critical = True
                if not frame_has_critical:
                    predicted_critical_noncritical += 1
        if not frame_has_critical:
            frames_without_critical += 1
            frames_with_false_brake += frame_has_predicted_critical
        for truth_index, truth in enumerate(frame.ground_truth):
            if classify_zone(truth[4]).zone == "critical_danger":
                critical_truth += 1
                if truth_index in truth_matches:
                    prediction_index = truth_matches[truth_index]
                    box = frame.predictions[prediction_index][0]
                    height = box[3] - box[1]
                    if height > 0 and classify_zone(621.0 * 1.7 / height).zone == "critical_danger":
                        matched_critical += 1
            else:
                noncritical_truth += 1
    return {
        "critical_ground_truth_objects": critical_truth,
        "critical_zone_recall": matched_critical / critical_truth if critical_truth else float("nan"),
        "noncritical_ground_truth_objects": noncritical_truth,
        "false_brake_rate_per_object": predicted_critical_noncritical / noncritical_truth if noncritical_truth else float("nan"),
        "noncritical_frames": frames_without_critical,
        "false_brake_rate_per_frame": frames_with_false_brake / frames_without_critical if frames_without_critical else float("nan"),
    }


def _save_audit_frames(frames: Sequence[FrameData], output_dir: Path) -> None:
    """Save 30 deterministic random and 30 worst-missed visual audit frames."""
    output_dir.mkdir(parents=True, exist_ok=True)
    random_frames = random.Random(42).sample(list(frames), min(30, len(frames)))
    missed_frames = sorted(
        frames,
        key=lambda frame: (
            -sum(
                1 for truth in frame.ground_truth
                if not any(iou(truth[:4], prediction[0]) >= 0.5 for prediction in frame.predictions)
            ),
            min((truth[3] - truth[1] for truth in frame.ground_truth), default=float("inf")),
        ),
    )[:30]
    for prefix, selected in (("random", random_frames), ("worst_missed", missed_frames)):
        for index, frame in enumerate(selected, start=1):
            image = cv2.imread(str(frame.image_path))
            if image is None:
                continue
            for truth in frame.ground_truth:
                cv2.rectangle(image, (int(truth[0]), int(truth[1])), (int(truth[2]), int(truth[3])), (0, 0, 255), 2)
            for prediction, confidence in frame.predictions:
                cv2.rectangle(image, (int(prediction[0]), int(prediction[1])), (int(prediction[2]), int(prediction[3])), (0, 255, 0), 2)
                cv2.putText(image, f"{confidence:.2f}", (int(prediction[0]), max(15, int(prediction[1]) - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            cv2.imwrite(str(output_dir / f"{prefix}_{index:02d}_{frame.image_path.stem}.jpg"), image)


def evaluate(weights: Path, image_dir: Path, labels_dir: Path, output_dir: Path, imgsz: int = 1280, confidence: float = 0.25) -> dict[str, object]:
    """Run the complete audit and write separate, reproducible artifacts."""
    model = YOLO(str(weights))
    output_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    frames: list[FrameData] = []
    timings: list[float] = []
    for index, image_path in enumerate(images, start=1):
        start = time.perf_counter()
        result = model.predict(source=str(image_path), imgsz=imgsz, conf=confidence, verbose=False)[0]
        timings.append(time.perf_counter() - start)
        predictions = [
            (box.xyxy[0].tolist(), float(box.conf[0]))
            for box in result.boxes if int(box.cls[0]) == 0
        ]
        frames.append(FrameData(image_path, _ground_truth(labels_dir / f"{image_path.stem}.txt"), predictions))
        if index % 500 == 0 or index == len(images):
            LOGGER.info("Audited %d/%d images", index, len(images))

    setup = {
        "iou_thresholds": [0.3, 0.5],
        "confidence_threshold": confidence,
        "imgsz": imgsz,
        "class_filter": "class_id == 0 (Pedestrian in yolo model)",
        "matching": "confidence-ordered one-to-one; each prediction and GT can match at most once",
        "geometric_depth": "Z = 621 * 1.7 / predicted_bbox_height_px",
        "centre_variant": "predicted box centre inside GT box; one-to-one, confidence ordered",
    }
    stats = _label_statistics(labels_dir)
    matching_rows = [
        _metrics(frames, threshold, mode)
        for mode in ("iou", "centre_inside")
        for threshold in (0.3, 0.5)
    ]
    stratified_rows = []
    for dimension, labels in (("distance", [name for _, _, name in DISTANCE_BINS]), ("occlusion", list(OCCLUSION_LEVELS)), ("height", [name for name, _, _ in HEIGHT_BINS])):
        for label in labels:
            row = _metrics(frames, 0.5, "iou", label, dimension)
            row["dimension"] = dimension
            stratified_rows.append(row)
    difficulty_rows = [_metrics(frames, 0.5, "iou", subset) for subset in ("easy", "moderate", "hard")]
    unfiltered = _metrics(frames, 0.5, "iou")
    critical = _critical_metrics(frames)
    report = {
        "matching_setup": setup,
        "label_statistics": stats,
        "unfiltered_iou_0.5": unfiltered,
        "matching_sweep": matching_rows,
        "stratified_results": stratified_rows,
        "difficulty_results": difficulty_rows,
        "critical_metrics_all_ground_truth": critical,
        "mean_inference_ms": float(np.mean(timings) * 1000),
        "images": len(images),
    }
    (output_dir / "audit_report.json").write_text(json.dumps(report, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    (output_dir / "label_statistics.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    for filename, rows in (("matching_sweep.csv", matching_rows), ("stratified_results.csv", stratified_rows), ("difficulty_results.csv", difficulty_rows)):
        with (output_dir / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
            writer.writeheader()
            writer.writerows(rows)
    _save_audit_frames(frames, output_dir / "frames")
    return report


def _print_table(title: str, rows: Iterable[dict[str, object]]) -> None:
    """Print compact audit tables."""
    print(f"\n{title}")
    for row in rows:
        print(" | ".join(f"{key}={value}" for key, value in row.items()))


def main() -> None:
    """Run and print the evaluator audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=Path("yolo11n.pt"))
    parser.add_argument("--images", type=Path, default=Path("dataset/images"))
    parser.add_argument("--labels", type=Path, default=Path("dataset/labels"))
    parser.add_argument("--output", type=Path, default=Path("outputs/audit"))
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    report = evaluate(args.weights, args.images, args.labels, args.output, args.imgsz, args.conf)
    print(json.dumps(report["matching_setup"], indent=2))
    print(json.dumps(report["label_statistics"], indent=2))
    _print_table("MATCHING SWEEP", report["matching_sweep"])
    _print_table("STRATIFIED RESULTS", report["stratified_results"])
    _print_table("DIFFICULTY RESULTS", report["difficulty_results"])
    print("\nCRITICAL METRICS")
    print(json.dumps(report["critical_metrics_all_ground_truth"], indent=2))


if __name__ == "__main__":
    main()

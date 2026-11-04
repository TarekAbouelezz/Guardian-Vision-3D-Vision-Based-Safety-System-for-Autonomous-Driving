"""Evaluate YOLO detection and geometric safety metrics on the full dataset."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from collections import Counter
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from finetune import DISTANCE_BINS, iou, parse_pedestrians
from src.risk import ZoneThresholds, classify_zone

LOGGER = logging.getLogger(__name__)


def evaluate(weights: Path, image_dir: Path, labels_dir: Path, output_dir: Path, imgsz: int = 1280) -> dict[str, object]:
    """Compute full-dataset detection, distance, and zone metrics."""
    model = YOLO(str(weights))
    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = ZoneThresholds()
    tp = fp = fn = frame_tp = frame_tn = frame_fp = frame_fn = 0
    inference_seconds: list[float] = []
    distance_rows: list[dict[str, float | str]] = []
    zone_rows: list[dict[str, str]] = []
    confusion: Counter[tuple[str, str]] = Counter()
    images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"})

    for index, image_path in enumerate(images, start=1):
        ground_truth = parse_pedestrians(labels_dir / f"{image_path.stem}.txt")
        start = time.perf_counter()
        prediction = model.predict(source=str(image_path), imgsz=imgsz, verbose=False)[0]
        inference_seconds.append(time.perf_counter() - start)
        predicted = [
            (box.xyxy[0].tolist(), float(box.conf[0]))
            for box in prediction.boxes
            if int(box.cls[0]) == 0
        ]
        matched_predictions: set[int] = set()
        matched_ground_truth: set[int] = set()
        for gt_index, gt_box in enumerate(ground_truth):
            best = max(
                ((iou(gt_box[:4], candidate[0]), candidate_index) for candidate_index, candidate in enumerate(predicted) if candidate_index not in matched_predictions),
                default=(0.0, -1),
            )
            if best[0] >= 0.5:
                matched_predictions.add(best[1])
                matched_ground_truth.add(gt_index)
                tp += 1
                predicted_box = predicted[best[1]][0]
                predicted_height = predicted_box[3] - predicted_box[1]
                if predicted_height <= 0:
                    continue
                estimate = 621.0 * 1.7 / predicted_height
                truth_distance = gt_box[4]
                distance_rows.append({"distance_bin_m": next(name for low, high, name in DISTANCE_BINS if low <= truth_distance < high), "ground_truth_distance": truth_distance, "estimated_distance": estimate})
                truth_zone = classify_zone(truth_distance, thresholds).zone
                predicted_zone = classify_zone(estimate, thresholds).zone
                confusion[(truth_zone, predicted_zone)] += 1
                zone_rows.append({"ground_truth_zone": truth_zone, "predicted_zone": predicted_zone})
        fp += len(predicted) - len(matched_predictions)
        fn += len(ground_truth) - len(matched_ground_truth)
        has_truth = bool(ground_truth)
        has_prediction = bool(predicted)
        frame_tp += has_truth and has_prediction
        frame_tn += not has_truth and not has_prediction
        frame_fp += not has_truth and has_prediction
        frame_fn += has_truth and not has_prediction
        if index % 500 == 0 or index == len(images):
            LOGGER.info("Evaluated %d/%d images", index, len(images))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    frame_accuracy = (frame_tp + frame_tn) / len(images) if images else 0.0
    truth = np.asarray([row["ground_truth_distance"] for row in distance_rows], dtype=float)
    estimate = np.asarray([row["estimated_distance"] for row in distance_rows], dtype=float)
    critical_truth = [row for row in zone_rows if row["ground_truth_zone"] == "critical_danger"]
    noncritical_truth = [row for row in zone_rows if row["ground_truth_zone"] != "critical_danger"]
    critical_recall = sum(row["predicted_zone"] == "critical_danger" for row in critical_truth) / len(critical_truth) if critical_truth else float("nan")
    false_brake_rate = sum(row["predicted_zone"] == "critical_danger" for row in noncritical_truth) / len(noncritical_truth) if noncritical_truth else float("nan")
    report = {
        "images": len(images),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "frame_presence_accuracy": frame_accuracy,
        "matched_distance_samples": len(distance_rows),
        "distance_mae_m": float(np.mean(np.abs(estimate - truth))) if len(truth) else float("nan"),
        "distance_absrel": float(np.mean(np.abs(estimate - truth) / np.maximum(truth, 1e-8))) if len(truth) else float("nan"),
        "critical_zone_recall": critical_recall,
        "false_brake_rate": false_brake_rate,
        "mean_inference_ms": float(np.mean(inference_seconds) * 1000) if inference_seconds else float("nan"),
        "zone_confusion": {f"{truth_zone}->{predicted_zone}": count for (truth_zone, predicted_zone), count in confusion.items()},
    }
    (output_dir / "full_evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    with (output_dir / "matched_distances.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=distance_rows[0].keys() if distance_rows else ["distance_bin_m", "ground_truth_distance", "estimated_distance"])
        writer.writeheader()
        writer.writerows(distance_rows)
    return report


def main() -> None:
    """Run full evaluation and print the summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=Path("yolo11n.pt"))
    parser.add_argument("--images", type=Path, default=Path("dataset/images"))
    parser.add_argument("--labels", type=Path, default=Path("dataset/labels"))
    parser.add_argument("--output", type=Path, default=Path("outputs/results/full"))
    parser.add_argument("--imgsz", type=int, default=1280)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    report = evaluate(args.weights, args.images, args.labels, args.output, args.imgsz)
    print(json.dumps(report, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()

"""Evaluate pedestrian detection with a ground-truth 3D corridor filter."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from ultralytics import YOLO

from finetune import DISTANCE_BINS, iou
from src.risk import classify_zone

LOGGER = logging.getLogger(__name__)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
PRIMARY_CORRIDOR_M = 2.0
SENSITIVITY_CORRIDORS_M = (1.5, 3.0)
FORWARD_LIMIT_M = 30.0


@dataclass(frozen=True)
class GroundTruth:
    """KITTI pedestrian geometry required by the evaluation."""

    bbox: tuple[float, float, float, float]
    lateral_m: float
    forward_m: float
    occlusion: int


@dataclass(frozen=True)
class FrameRecord:
    """Per-frame evaluation inputs and aggregateable outcomes."""

    frame_id: str
    sequence_id: str
    ground_truth: tuple[GroundTruth, ...]
    predictions: tuple[tuple[tuple[float, float, float, float], float], ...]


def in_corridor(ground_truth: GroundTruth, lateral_limit_m: float, forward_limit_m: float = FORWARD_LIMIT_M) -> bool:
    """Return whether a ground-truth pedestrian belongs to a corridor."""
    return abs(ground_truth.lateral_m) <= lateral_limit_m and ground_truth.forward_m <= forward_limit_m


def _parse_ground_truth(label_path: Path) -> list[GroundTruth]:
    """Read pedestrian KITTI rows, including lateral and forward position."""
    if not label_path.exists():
        return []
    result = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 15 or fields[0].lower() != "pedestrian":
            continue
        try:
            values = [float(value) for value in fields[1:]]
            result.append(GroundTruth(
                bbox=(values[3], values[4], values[5], values[6]),
                lateral_m=values[10],
                forward_m=values[12],
                occlusion=int(values[1]),
            ))
        except (TypeError, ValueError):
            continue
    return result


def _metadata_sequences(metadata_path: Path) -> dict[str, str]:
    """Map six-digit frame IDs to run/scenario sequence IDs."""
    import csv as csv_module

    with metadata_path.open(newline="", encoding="utf-8") as handle:
        return {
            f"{int(row['frame_id']):06d}": f"{row['run_id']}::{row['scenario']}"
            for row in csv_module.DictReader(handle)
        }


def _match(
    ground_truth: Sequence[GroundTruth],
    predictions: Sequence[tuple[tuple[float, float, float, float], float]],
    threshold: float = 0.5,
) -> tuple[dict[int, int], dict[int, int]]:
    """Confidence-ordered one-to-one IoU matching."""
    prediction_matches: dict[int, int] = {}
    truth_matches: dict[int, int] = {}
    for prediction_index in sorted(range(len(predictions)), key=lambda index: -predictions[index][1]):
        candidates = [
            (iou(predictions[prediction_index][0], truth.bbox), truth_index)
            for truth_index, truth in enumerate(ground_truth)
            if truth_index not in truth_matches
        ]
        if candidates:
            score, truth_index = max(candidates)
            if score >= threshold:
                prediction_matches[prediction_index] = truth_index
                truth_matches[truth_index] = prediction_index
    return prediction_matches, truth_matches


def match_with_ignore_region(
    included_ground_truth: Sequence[GroundTruth],
    ignored_ground_truth: Sequence[GroundTruth],
    predictions: Sequence[tuple[tuple[float, float, float, float], float]],
    threshold: float = 0.5,
) -> tuple[int, int, int, tuple[tuple[int, int], ...]]:
    """Match included GT, then ignore predictions matching outside-region GT."""
    included_prediction_matches, included_truth_matches = _match(included_ground_truth, predictions, threshold)
    remaining_predictions = [
        (index, prediction)
        for index, prediction in enumerate(predictions)
        if index not in included_prediction_matches
    ]
    ignored_prediction_matches = _match(
        ignored_ground_truth,
        [prediction for _, prediction in remaining_predictions],
        threshold,
    )[0]
    ignored_prediction_indices = {
        remaining_predictions[prediction_index][0]
        for prediction_index in ignored_prediction_matches
    }
    false_positives = len(predictions) - len(included_prediction_matches) - len(ignored_prediction_indices)
    false_negatives = len(included_ground_truth) - len(included_truth_matches)
    pairs = tuple(sorted(included_prediction_matches.items()))
    return len(included_truth_matches), false_positives, false_negatives, pairs


def _prediction_distance(prediction: tuple[float, float, float, float]) -> float | None:
    """Estimate distance from a predicted box height."""
    height = prediction[3] - prediction[1]
    return 621.0 * 1.7 / height if height > 0 else None


def _view_frame_metrics(frame: FrameRecord, corridor_limit_m: float | None) -> dict[str, float | int]:
    """Compute additive per-frame metrics for one view."""
    if corridor_limit_m is None:
        included = list(frame.ground_truth)
        ignored: list[GroundTruth] = []
    else:
        included = [truth for truth in frame.ground_truth if in_corridor(truth, corridor_limit_m)]
        ignored = [truth for truth in frame.ground_truth if not in_corridor(truth, corridor_limit_m)]
    tp, fp, fn, pairs = match_with_ignore_region(included, ignored, frame.predictions)
    errors: list[float] = []
    absrel: list[float] = []
    critical_total = sum(classify_zone(truth.forward_m).zone == "critical_danger" for truth in frame.ground_truth)
    critical_detected = 0
    for prediction_index, truth_index in pairs:
        distance = _prediction_distance(frame.predictions[prediction_index][0])
        if distance is None:
            continue
        truth = included[truth_index]
        errors.append(abs(distance - truth.forward_m))
        absrel.append(abs(distance - truth.forward_m) / max(truth.forward_m, 1e-8))
        if classify_zone(distance).zone == "critical_danger" and classify_zone(truth.forward_m).zone == "critical_danger":
            critical_detected += 1
    predicted_critical = sum(
        _prediction_distance(prediction[0]) is not None
        and classify_zone(_prediction_distance(prediction[0]) or 0).zone == "critical_danger"
        for prediction in frame.predictions
    )
    frame_has_critical = any(classify_zone(truth.forward_m).zone == "critical_danger" for truth in frame.ground_truth)
    false_brake_objects = 0
    matched_prediction_indices = {prediction_index for prediction_index, _ in pairs}
    for prediction_index, (prediction, _) in enumerate(frame.predictions):
        distance = _prediction_distance(prediction)
        if distance is None or classify_zone(distance).zone != "critical_danger":
            continue
        if not frame_has_critical:
            false_brake_objects += 1
        elif prediction_index in matched_prediction_indices:
            truth_index = next(
                pair_truth_index
                for pair_prediction_index, pair_truth_index in pairs
                if pair_prediction_index == prediction_index
            )
            if classify_zone(included[truth_index].forward_m).zone != "critical_danger":
                false_brake_objects += 1
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "distance_error_sum": float(sum(errors)),
        "absrel_sum": float(sum(absrel)),
        "distance_count": len(errors),
        "critical_total": critical_total,
        "critical_detected": critical_detected,
        "predicted_critical": predicted_critical,
        "false_brake_objects": false_brake_objects,
        "frame_has_critical": int(frame_has_critical),
        "frame_false_brake": int(not frame_has_critical and predicted_critical > 0),
        "included_ground_truth": len(included),
        "empty_frame": int(not included),
    }


def _aggregate(frame_metrics: Sequence[dict[str, float | int]]) -> dict[str, float | int]:
    """Aggregate additive frame metrics into the requested view metrics."""
    sums = {key: sum(float(row[key]) for row in frame_metrics) for key in frame_metrics[0]} if frame_metrics else {}
    tp, fp, fn = sums.get("tp", 0), sums.get("fp", 0), sums.get("fn", 0)
    critical_total = sums.get("critical_total", 0)
    noncritical_objects = sum(
        float(row["included_ground_truth"]) - float(row["critical_total"])
        for row in frame_metrics
    )
    no_critical_frames = sum(1 - int(row["frame_has_critical"]) for row in frame_metrics)
    return {
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "distance_mae_m": sums.get("distance_error_sum", 0.0) / sums.get("distance_count", 1),
        "distance_absrel": sums.get("absrel_sum", 0.0) / sums.get("distance_count", 1),
        "critical_zone_recall": sums.get("critical_detected", 0.0) / critical_total if critical_total else float("nan"),
        "false_brake_rate_per_object": sums.get("false_brake_objects", 0.0) / noncritical_objects if noncritical_objects else float("nan"),
        "false_brake_rate_per_frame": sums.get("frame_false_brake", 0.0) / no_critical_frames if no_critical_frames else float("nan"),
    }


def _bootstrap(
    frame_metrics_by_sequence: dict[str, list[dict[str, float | int]]],
    iterations: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Compute percentile 95% CIs by resampling complete sequences."""
    sequence_ids = sorted(frame_metrics_by_sequence)
    rng = random.Random(seed)
    samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(iterations):
        selected = [rng.choice(sequence_ids) for _ in sequence_ids]
        rows = [row for sequence_id in selected for row in frame_metrics_by_sequence[sequence_id]]
        aggregate = _aggregate(rows)
        for metric in ("precision", "recall", "f1", "distance_mae_m", "distance_absrel", "critical_zone_recall", "false_brake_rate_per_frame", "false_brake_rate_per_object"):
            value = float(aggregate[metric])
            if np.isfinite(value):
                samples[metric].append(value)
    return {
        metric: {
            "lower_95": float(np.percentile(values, 2.5)),
            "upper_95": float(np.percentile(values, 97.5)),
        }
        for metric, values in samples.items()
    }


def _empty_frame_metrics(frames: Sequence[FrameRecord]) -> dict[str, float | int]:
    """Report false alarms and frame-level metrics on labelled empty frames."""
    empty = [frame for frame in frames if not frame.ground_truth]
    false_positives = sum(len(frame.predictions) for frame in empty)
    false_alarm_frames = sum(bool(frame.predictions) for frame in empty)
    nonempty = [frame for frame in frames if frame.ground_truth]
    frame_tp = sum(bool(frame.predictions) for frame in nonempty)
    frame_fn = len(nonempty) - frame_tp
    frame_tn = len(empty) - false_alarm_frames
    frame_fp = false_alarm_frames
    return {
        "empty_frames": len(empty),
        "false_positives": false_positives,
        "false_positives_per_empty_frame": false_positives / len(empty) if empty else float("nan"),
        "empty_frames_with_false_alarm_percent": 100.0 * false_alarm_frames / len(empty) if empty else float("nan"),
        "frame_precision": frame_tp / (frame_tp + frame_fp) if frame_tp + frame_fp else float("nan"),
        "frame_recall": frame_tp / (frame_tp + frame_fn) if frame_tp + frame_fn else float("nan"),
        "frame_specificity": frame_tn / (frame_tn + frame_fp) if frame_tn + frame_fp else float("nan"),
    }


def _write_rows(path: Path, rows: Sequence[dict[str, object]]) -> None:
    """Write dictionaries with a stable union of columns."""
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(
    weights: Path,
    image_dir: Path,
    labels_dir: Path,
    metadata_path: Path,
    output_dir: Path,
    imgsz: int = 1280,
    confidence: float = 0.25,
    bootstrap_iterations: int = 1000,
    bootstrap_seed: int = 42,
) -> dict[str, object]:
    """Run corridor and all-pedestrian evaluation into the v2 directory."""
    model = YOLO(str(weights))
    output_dir.mkdir(parents=True, exist_ok=True)
    sequence_map = _metadata_sequences(metadata_path)
    images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    frames: list[FrameRecord] = []
    timings: list[float] = []
    for index, image_path in enumerate(images, start=1):
        start = time.perf_counter()
        result = model.predict(source=str(image_path), imgsz=imgsz, conf=confidence, verbose=False)[0]
        timings.append(time.perf_counter() - start)
        predictions = tuple(
            (tuple(float(value) for value in box.xyxy[0].tolist()), float(box.conf[0]))
            for box in result.boxes if int(box.cls[0]) == 0
        )
        frames.append(FrameRecord(
            image_path.stem,
            sequence_map.get(image_path.stem, f"unknown::{image_path.stem}"),
            tuple(_parse_ground_truth(labels_dir / f"{image_path.stem}.txt")),
            predictions,
        ))
        if index % 500 == 0 or index == len(images):
            LOGGER.info("Evaluated %d/%d images", index, len(images))

    views: dict[str, object] = {}
    for name, corridor_limit in (
        ("all_pedestrians", None),
        ("corridor_2.0m", PRIMARY_CORRIDOR_M),
        ("corridor_1.5m", SENSITIVITY_CORRIDORS_M[0]),
        ("corridor_3.0m", SENSITIVITY_CORRIDORS_M[1]),
    ):
        per_frame = [_view_frame_metrics(frame, corridor_limit) for frame in frames]
        by_sequence: dict[str, list[dict[str, float | int]]] = defaultdict(list)
        for frame, metrics in zip(frames, per_frame):
            by_sequence[frame.sequence_id].append(metrics)
        aggregate = _aggregate(per_frame)
        aggregate["ground_truth_pedestrians"] = int(sum(row["included_ground_truth"] for row in per_frame))
        aggregate["frames_with_included_ground_truth"] = int(sum(bool(row["included_ground_truth"]) for row in per_frame))
        aggregate["bootstrap_ci_95"] = _bootstrap(by_sequence, bootstrap_iterations, bootstrap_seed)
        views[name] = aggregate

    empty_frames = _empty_frame_metrics(frames)
    report = {
        "protocol": {
            "primary_corridor": "|lateral| <= 2.0m and forward <= 30m",
            "sensitivity_corridors": ["|lateral| <= 1.5m and forward <= 30m", "|lateral| <= 3.0m and forward <= 30m"],
            "ignored_outside_corridor": True,
            "confidence_threshold": confidence,
            "imgsz": imgsz,
            "class_filter": "class_id == 0",
            "matching": "confidence-ordered one-to-one IoU >= 0.5; outside-region matches ignored",
            "depth": "621 * 1.7 / predicted box height",
            "bootstrap": f"{bootstrap_iterations} iterations, sequence-resampled, seed={bootstrap_seed}",
        },
        "images": len(frames),
        "empty_frames": empty_frames,
        "views": views,
        "mean_inference_ms": float(np.mean(timings) * 1000) if timings else float("nan"),
    }
    (output_dir / "corridor_report.json").write_text(json.dumps(report, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    _write_rows(output_dir / "view_metrics.csv", [
        {"view": name, **{key: value for key, value in values.items() if key != "bootstrap_ci_95"}}
        for name, values in views.items()
    ])
    _write_rows(output_dir / "empty_frame_metrics.csv", [{"view": "empty_frames", **empty_frames}])
    return report


def main() -> None:
    """Run corridor evaluation and print the three requested views."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=Path("yolo11n.pt"))
    parser.add_argument("--images", type=Path, default=Path("dataset/images"))
    parser.add_argument("--labels", type=Path, default=Path("dataset/labels"))
    parser.add_argument("--metadata", type=Path, default=Path("dataset/metadata.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/results_v2"))
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    report = evaluate(args.weights, args.images, args.labels, args.metadata, args.output, args.imgsz, args.conf, args.bootstrap_iterations)
    print(f"Empty frames: {report['empty_frames']['empty_frames']}")
    for name, values in report["views"].items():
        print(f"{name}: ground_truth={values['ground_truth_pedestrians']} frames={values['frames_with_included_ground_truth']}")
    for name in ("all_pedestrians", "corridor_2.0m", "empty_frames"):
        print(f"\n{name}")
        print(json.dumps(report["views"].get(name, report["empty_frames"]), indent=2, allow_nan=True))


if __name__ == "__main__":
    main()

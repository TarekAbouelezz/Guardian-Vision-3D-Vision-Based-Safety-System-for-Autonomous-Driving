"""Inference-only test evaluation for pretrained and existing fine-tuned YOLO models."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

from finetune import DISTANCE_BINS, iou

LOGGER = logging.getLogger(__name__)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
FX = FY = 621.0
CORRIDOR_LATERAL_M = 2.0
CORRIDOR_FORWARD_M = 30.0
CRITICAL_DISTANCE_M = 7.0
WARNING_DISTANCE_M = 15.0
IOU_THRESHOLD = 0.5
CONFIDENCE_THRESHOLD = 0.25
BOOTSTRAP_SEED = 20261109
BOOTSTRAP_REPLICATES = 2000


@dataclass(frozen=True)
class GroundTruth:
    bbox: tuple[float, float, float, float]
    lateral_m: float
    forward_m: float
    occlusion: int


@dataclass(frozen=True)
class Match:
    truth: GroundTruth
    prediction: tuple[tuple[float, float, float, float], float]


@dataclass
class FrameResult:
    frame_id: str
    sequence_id: str
    ground_truth: list[GroundTruth]
    predictions: list[tuple[tuple[float, float, float, float], float]]
    matches: list[Match]
    false_positives: int
    inference_ms: float


def parse_ground_truth(path: Path) -> list[GroundTruth]:
    """Read pedestrian boxes and 3D position from a KITTI annotation."""
    if not path.exists():
        return []
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 15 or fields[0].lower() != "pedestrian":
            continue
        values = [float(value) for value in fields[1:]]
        result.append(GroundTruth(
            bbox=(values[3], values[4], values[5], values[6]),
            lateral_m=values[10],
            forward_m=values[12],
            occlusion=int(values[1]),
        ))
    return result


def load_test_rows(
    manifest: Path,
    metadata: Path,
    image_root: Path,
) -> list[tuple[str, Path, str]]:
    """Load exactly the frame IDs listed by the generated test manifest."""
    sequence_by_frame: dict[str, str] = {}
    with metadata.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            sequence_by_frame[f"{int(row['frame_id']):06d}"] = f"{row['run_id']}::{row['scenario']}"
    rows = []
    for entry in manifest.read_text(encoding="utf-8").splitlines():
        if not entry.strip():
            continue
        frame_id = Path(entry.strip()).stem
        image_path = image_root / f"{frame_id}.png"
        if not image_path.exists():
            candidates = list(image_root.glob(f"{frame_id}.*"))
            if not candidates:
                raise FileNotFoundError(f"Test image missing for frame {frame_id} in {image_root}")
            image_path = candidates[0]
        if not image_path.exists():
            raise FileNotFoundError(f"Test image missing: {image_path}")
        if frame_id not in sequence_by_frame:
            raise KeyError(f"No metadata sequence for test frame {frame_id}")
        rows.append((frame_id, image_path, sequence_by_frame[frame_id]))
    return rows


def predict_model(model: YOLO, rows: Sequence[tuple[str, Path, str]], labels_dir: Path, imgsz: int) -> list[FrameResult]:
    """Run inference and confidence-ordered one-to-one IoU matching."""
    results: list[FrameResult] = []
    for index, (frame_id, image_path, sequence_id) in enumerate(rows, start=1):
        ground_truth = parse_ground_truth(labels_dir / f"{frame_id}.txt")
        started = time.perf_counter()
        prediction = model.predict(
            source=str(image_path),
            imgsz=imgsz,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False,
        )[0]
        inference_ms = (time.perf_counter() - started) * 1000.0
        predictions = [
            (tuple(float(value) for value in box.xyxy[0].tolist()), float(box.conf[0]))
            for box in prediction.boxes
            if int(box.cls[0]) == 0
        ]
        used_predictions: set[int] = set()
        matches: list[Match] = []
        for truth in ground_truth:
            candidates = [
                (iou(truth.bbox, candidate[0]), candidate_index)
                for candidate_index, candidate in enumerate(predictions)
                if candidate_index not in used_predictions
            ]
            if candidates:
                score, candidate_index = max(candidates)
                if score >= IOU_THRESHOLD:
                    used_predictions.add(candidate_index)
                    matches.append(Match(truth, predictions[candidate_index]))
        results.append(FrameResult(
            frame_id=frame_id,
            sequence_id=sequence_id,
            ground_truth=ground_truth,
            predictions=predictions,
            matches=matches,
            false_positives=len(predictions) - len(used_predictions),
            inference_ms=inference_ms,
        ))
        if index % 100 == 0 or index == len(rows):
            LOGGER.info("Evaluated %d/%d frames", index, len(rows))
    return results


def distance_from_prediction(prediction: tuple[float, float, float, float]) -> float | None:
    """Estimate metric distance from predicted bbox height."""
    height = prediction[3] - prediction[1]
    return FY * 1.7 / height if height > 0 else None


def prediction_position(prediction: tuple[float, float, float, float]) -> tuple[float, float] | None:
    """Return predicted lateral and forward metres from a geometric box."""
    distance = distance_from_prediction(prediction)
    if distance is None:
        return None
    center_x = (prediction[0] + prediction[2]) / 2.0
    return ((center_x - 621.0) * distance / FX, distance)


def in_corridor(truth: GroundTruth) -> bool:
    """Return whether a ground-truth pedestrian is in the 2 m ego corridor."""
    return abs(truth.lateral_m) <= CORRIDOR_LATERAL_M and truth.forward_m <= CORRIDOR_FORWARD_M


def prediction_in_corridor(prediction: tuple[float, float, float, float]) -> bool:
    """Return whether a predicted box projects into the 2 m ego corridor."""
    position = prediction_position(prediction)
    return position is not None and abs(position[0]) <= CORRIDOR_LATERAL_M and position[1] <= CORRIDOR_FORWARD_M


def zone(distance: float) -> str:
    """Classify a metric distance using the configured safety thresholds."""
    if distance < CRITICAL_DISTANCE_M:
        return "critical_danger"
    if distance < WARNING_DISTANCE_M:
        return "warning"
    return "monitoring"


def _metric_counts(
    frames: Sequence[FrameResult],
    corridor_only: bool = False,
    distance_range: tuple[float, float] | None = None,
) -> dict[str, float]:
    """Compute metrics after applying identical GT/prediction subset filters."""
    tp = fp = fn = 0
    distance_errors: list[tuple[float, float, float]] = []
    critical_total = critical_detected = false_brake_objects = 0
    false_brake_frames = 0
    frame_denominator = 0
    frame_presence_correct = 0
    for frame in frames:
        included_truth = [
            truth for truth in frame.ground_truth
            if (not corridor_only or in_corridor(truth))
            and (distance_range is None or distance_range[0] <= truth.forward_m < distance_range[1])
        ]
        if corridor_only:
            included_predictions = [prediction for prediction in frame.predictions if prediction_in_corridor(prediction[0])]
        elif distance_range is not None:
            included_predictions = [
                prediction for prediction in frame.predictions
                if distance_from_prediction(prediction[0]) is not None
                and distance_range[0] <= (distance_from_prediction(prediction[0]) or 0.0) < distance_range[1]
            ]
        else:
            included_predictions = list(frame.predictions)
        matched_prediction_indices, matched_truth_indices = match_subset(included_truth, included_predictions)
        included_matches = [
            Match(included_truth[truth_index], included_predictions[prediction_index])
            for prediction_index, truth_index in matched_prediction_indices.items()
        ]
        tp += len(included_matches)
        fn += len(included_truth) - len(matched_truth_indices)
        fp += len(included_predictions) - len(matched_prediction_indices)
        for match in included_matches:
            estimated = distance_from_prediction(match.prediction[0])
            if estimated is None:
                continue
            distance_errors.append((match.truth.forward_m, estimated, abs(estimated - match.truth.forward_m) / max(match.truth.forward_m, 1e-8)))
        critical_truth = [truth for truth in included_truth if zone(truth.forward_m) == "critical_danger"]
        critical_total += len(critical_truth)
        critical_detected += sum(
            zone(distance_from_prediction(match.prediction[0]) or float("inf")) == "critical_danger"
            for match in included_matches
            if match.truth in critical_truth
        )
        predicted_critical_indices = {
            index for index, prediction in enumerate(included_predictions)
            if zone(distance_from_prediction(prediction[0]) or float("inf")) == "critical_danger"
        }
        matched_critical_indices = {
            prediction_index
            for prediction_index, truth_index in matched_prediction_indices.items()
            if zone(included_truth[truth_index].forward_m) == "critical_danger"
        }
        false_brake_objects += sum(index not in matched_critical_indices for index in predicted_critical_indices)
        frame_has_critical = bool(critical_truth)
        frame_predicted_critical = bool(predicted_critical_indices)
        frame_denominator += 1
        frame_presence_correct += bool(included_truth) == bool(included_predictions)
        false_brake_frames += int(frame_predicted_critical and not frame_has_critical)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    distance_values = np.asarray(distance_errors, dtype=float)
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "distance_mae_m": float(np.mean(distance_values[:, 2] * 0 + np.abs(distance_values[:, 1] - distance_values[:, 0]))) if len(distance_values) else float("nan"),
        "distance_absrel": float(np.mean(distance_values[:, 2])) if len(distance_values) else float("nan"),
        "critical_zone_recall": critical_detected / critical_total if critical_total else float("nan"),
        "false_brake_per_object": false_brake_objects / max(
            sum(
                len([
                    prediction for prediction in frame.predictions
                    if corridor_only and prediction_in_corridor(prediction[0])
                    or distance_range is None and not corridor_only
                    or distance_range is not None and distance_from_prediction(prediction[0]) is not None
                    and distance_range[0] <= (distance_from_prediction(prediction[0]) or 0.0) < distance_range[1]
                ])
                for frame in frames
            ),
            1,
        ),
        "false_brake_per_frame": false_brake_frames / max(frame_denominator, 1),
        "mean_inference_ms": float(np.mean([frame.inference_ms for frame in frames])) if frames else float("nan"),
        "frame_presence_accuracy": frame_presence_correct / max(frame_denominator, 1),
    }


def match_subset(
    ground_truth: Sequence[GroundTruth],
    predictions: Sequence[tuple[tuple[float, float, float, float], float]],
) -> tuple[dict[int, int], dict[int, int]]:
    """Match confidence-ordered predictions to a filtered GT subset."""
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
            if score >= IOU_THRESHOLD:
                prediction_matches[prediction_index] = truth_index
                truth_matches[truth_index] = prediction_index
    return prediction_matches, truth_matches


def frame_predictions(frames: Sequence[FrameResult], corridor_only: bool) -> list[tuple[tuple[float, float, float, float], float]]:
    """Return predictions eligible for the requested report."""
    predictions = []
    for frame in frames:
        ignored = {index for index, prediction in enumerate(frame.predictions) if corridor_only and any(iou(prediction[0], truth.bbox) >= IOU_THRESHOLD for truth in frame.ground_truth if not in_corridor(truth))}
        predictions.extend(prediction for index, prediction in enumerate(frame.predictions) if index not in ignored)
    return predictions


def _bin_frames(frames: Sequence[FrameResult], low: float, high: float) -> list[FrameResult]:
    """Keep frames containing at least one GT object in a distance bin."""
    return [frame for frame in frames if any(low <= truth.forward_m < high for truth in frame.ground_truth)]


def bootstrap_ci(frames: Sequence[FrameResult], metric_fn, replicates: int = BOOTSTRAP_REPLICATES) -> tuple[float, float]:
    """Bootstrap a scalar metric by resampling whole sequences."""
    groups: dict[str, list[FrameResult]] = defaultdict(list)
    for frame in frames:
        groups[frame.sequence_id].append(frame)
    sequences = list(groups.values())
    if not sequences:
        return float("nan"), float("nan")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values = []
    for _ in range(replicates):
        sampled = [sequences[index] for index in rng.integers(0, len(sequences), len(sequences))]
        values.append(metric_fn([frame for group in sampled for frame in group]))
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def summarize(frames: Sequence[FrameResult], label: str) -> dict[str, object]:
    """Create all/corridor/bin metrics and sequence-bootstrap confidence intervals."""
    reports: dict[str, object] = {}
    subsets: dict[str, tuple[list[FrameResult], bool, tuple[float, float] | None]] = {
        "all_pedestrians": (frames, False, None),
        "corridor_2.0m": (frames, True, None),
    }
    for low, high, name in DISTANCE_BINS:
        subsets[f"distance_{name}m"] = (frames, False, (low, high))
    for name, (subset, corridor_only, distance_range) in subsets.items():
        metrics = _metric_counts(subset, corridor_only=corridor_only, distance_range=distance_range)
        metrics["bootstrap_95_ci"] = {
            metric: bootstrap_ci(
                subset,
                lambda sampled, metric=metric: _metric_counts(
                    sampled,
                    corridor_only=corridor_only,
                    distance_range=distance_range,
                )[metric],
            )
            for metric in ("precision", "recall", "f1", "distance_mae_m", "distance_absrel", "critical_zone_recall", "false_brake_per_object", "false_brake_per_frame")
        }
        reports[name] = metrics
    return {"model": label, "confidence_threshold": CONFIDENCE_THRESHOLD, "imgsz": 1280, "iou_threshold": IOU_THRESHOLD, "metrics": reports}


def write_rows(output_dir: Path, model_name: str, report: dict[str, object], frames: Sequence[FrameResult]) -> None:
    """Write JSON and frame-level records for reproducibility."""
    (output_dir / f"{model_name}_report.json").write_text(json.dumps(report, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    with (output_dir / f"{model_name}_frames.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_id", "sequence_id", "ground_truth_objects", "predicted_objects", "matches", "false_positives", "inference_ms"])
        for frame in frames:
            writer.writerow([frame.frame_id, frame.sequence_id, len(frame.ground_truth), len(frame.predictions), len(frame.matches), frame.false_positives, frame.inference_ms])


def save_training_curves(training_dir: Path, output_dir: Path) -> None:
    """Copy existing training artifacts and plot loss/mAP curves without retraining."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("results.png", "results.csv"):
        source = training_dir / name
        if source.exists():
            shutil.copy2(source, output_dir / name)
    results_csv = training_dir / "results.csv"
    if not results_csv.exists():
        return
    rows = list(csv.DictReader(results_csv.open(encoding="utf-8")))
    epochs = [int(row["epoch"]) for row in rows]
    train_box = [float(row["train/box_loss"]) for row in rows]
    val_box = [float(row["val/box_loss"]) for row in rows]
    map50 = [float(row["metrics/mAP50(B)"]) for row in rows]
    map5095 = [float(row["metrics/mAP50-95(B)"]) for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, train_box, label="train box loss")
    axes[0].plot(epochs, val_box, label="validation box loss")
    axes[0].set_title("Loss curves")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[1].plot(epochs, map50, label="validation mAP50")
    axes[1].plot(epochs, map5095, label="validation mAP50-95")
    axes[1].set_title("Validation mAP curves")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_dir / "loss_and_val_map_curves.png", dpi=160)
    plt.close(figure)
    best_index = int(np.argmax(map50))
    gap = {
        "best_epoch_by_val_mAP50": epochs[best_index],
        "best_val_mAP50": map50[best_index],
        "best_val_mAP50_95": map5095[best_index],
        "final_train_box_loss": train_box[-1],
        "final_val_box_loss": val_box[-1],
        "final_box_loss_gap_val_minus_train": val_box[-1] - train_box[-1],
    }
    (output_dir / "train_validation_gap.json").write_text(json.dumps(gap, indent=2) + "\n", encoding="utf-8")


def condition_summary(
    metadata_path: Path,
    train_manifest: Path,
    val_manifest: Path,
    test_manifest: Path,
) -> dict[str, dict[str, object]]:
    """Report weather/day-night composition for each split."""
    metadata = {}
    with metadata_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            metadata[f"{int(row['frame_id']):06d}"] = row

    def frame_ids(path: Path) -> set[str]:
        return {Path(line.strip()).stem for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}

    result = {}
    for name, manifest in (("train", train_manifest), ("validation", val_manifest), ("test", test_manifest)):
        rows = [metadata[frame_id] for frame_id in frame_ids(manifest) if frame_id in metadata]
        weather = Counter(row["weather"] for row in rows)
        day_night = Counter("night" if "night" in row["weather"].lower() else "day_or_unspecified" for row in rows)
        result[name] = {
            "frames": len(rows),
            "weather": dict(sorted(weather.items())),
            "day_night": dict(sorted(day_night.items())),
            "scenarios": dict(sorted(Counter(row["scenario"] for row in rows).items())),
            "sequences": len({row["run_id"] for row in rows}),
        }
    return result


def main() -> None:
    """Run inference-only test evaluation for both checkpoints."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--best", type=Path, default=Path("runs/detect/outputs/training/guardian_yolo11n-3/weights/best.pt"))
    parser.add_argument("--pretrained", type=Path, default=Path("yolo11n.pt"))
    parser.add_argument("--data", type=Path, default=Path("dataset/yolo/carla.yaml"))
    parser.add_argument("--test-manifest", type=Path, default=Path("dataset/yolo/test.txt"))
    parser.add_argument("--images", type=Path, default=Path("dataset/images"))
    parser.add_argument("--labels", type=Path, default=Path("dataset/labels"))
    parser.add_argument("--metadata", type=Path, default=Path("dataset/metadata.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/results_v3"))
    parser.add_argument("--imgsz", type=int, default=1280)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    if not args.best.exists() or not args.pretrained.exists():
        raise FileNotFoundError("Both --best and --pretrained checkpoints must exist")
    rows = load_test_rows(args.test_manifest, args.metadata, args.images)
    if len(rows) != 994:
        raise ValueError(f"Expected 994 test frames, found {len(rows)}")
    args.output.mkdir(parents=True, exist_ok=True)
    metadata_report = {
        "scope": "TEST ONLY",
        "frames": len(rows),
        "warning": "All 6,567-frame audits include training data and are not generalisation results.",
        "confidence_selection": "0.25 carried forward from validation audit; no test tuning.",
        "matching": "confidence-ordered one-to-one IoU >= 0.5",
        "depth": "geometric Z = 621 * 1.7 / predicted_bbox_height_px",
        "checkpoints_unchanged": True,
    }
    (args.output / "README.txt").write_text(
        "TEST-ONLY RESULTS. 994 frames. No retraining. Existing checkpoints used unchanged.\n"
        "Prior full-dataset audit/corridor outputs cover ALL 6,567 FRAMES, INCLUDES TRAINING DATA: not a generalisation result.\n",
        encoding="utf-8",
    )
    (args.output / "evaluation_protocol.json").write_text(json.dumps(metadata_report, indent=2) + "\n", encoding="utf-8")
    (args.output / "condition_summary.json").write_text(
        json.dumps(
            condition_summary(
                args.metadata,
                Path("dataset/yolo/train.txt"),
                Path("dataset/yolo/val.txt"),
                args.test_manifest,
            ),
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    for label, checkpoint in (("pretrained", args.pretrained), ("finetuned_best", args.best)):
        LOGGER.info("Running %s inference-only test evaluation", label)
        frames = predict_model(YOLO(str(checkpoint)), rows, args.labels, args.imgsz)
        write_rows(args.output, label, summarize(frames, label), frames)
    save_training_curves(
        args.best.parent.parent,
        args.output / "training_curves",
    )
    LOGGER.info("Running Ultralytics val(split=test) for both checkpoints")
    for label, checkpoint in (("pretrained", args.pretrained), ("finetuned_best", args.best)):
        model = YOLO(str(checkpoint))
        model.val(
            data=str(args.data),
            split="test",
            imgsz=args.imgsz,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            project=str(args.output / "ultralytics_val"),
            name=label,
            exist_ok=True,
            plots=True,
            verbose=False,
        )


if __name__ == "__main__":
    main()

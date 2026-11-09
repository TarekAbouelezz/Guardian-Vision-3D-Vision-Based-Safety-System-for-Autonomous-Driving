"""Diagnose critical-zone recall without changing v1 or v2 result artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
from ultralytics import YOLO

from evaluate_corridor import FORWARD_LIMIT_M, PRIMARY_CORRIDOR_M, in_corridor
from finetune import iou
from src.risk import classify_zone

LOGGER = logging.getLogger(__name__)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
FX = FY = 621.0
CX = 621.0
CY = 187.5
PERSON_HEIGHT_M = 1.7


@dataclass(frozen=True)
class Truth:
    bbox: tuple[float, float, float, float]
    truncation: float
    occlusion: int
    lateral_m: float
    forward_m: float


@dataclass(frozen=True)
class Frame:
    image_path: Path
    width: int
    height: int
    truths: tuple[Truth, ...]
    predictions: tuple[tuple[tuple[float, float, float, float], float], ...]


def _parse_truth(path: Path) -> tuple[Truth, ...]:
    result: list[Truth] = []
    if not path.exists():
        return ()
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 15 or fields[0].lower() != "pedestrian":
            continue
        try:
            values = [float(value) for value in fields[1:]]
            result.append(Truth(
                bbox=(values[3], values[4], values[5], values[6]),
                truncation=values[0],
                occlusion=int(values[1]),
                lateral_m=values[10],
                forward_m=values[12],
            ))
        except ValueError:
            continue
    return tuple(result)


def _clip_box(box: Sequence[float], width: int, height: int) -> tuple[float, float, float, float]:
    return (
        max(0.0, min(float(width), float(box[0]))),
        max(0.0, min(float(height), float(box[1]))),
        max(0.0, min(float(width), float(box[2]))),
        max(0.0, min(float(height), float(box[3]))),
    )


def _match(
    truths: Sequence[Truth],
    predictions: Sequence[tuple[tuple[float, float, float, float], float]],
    clip: bool = False,
    width: int = 1242,
    height: int = 375,
    prediction_order: bool = True,
) -> tuple[dict[int, int], dict[int, int]]:
    """Return one-to-one matches, using v2 confidence order by default."""
    truth_boxes = [
        _clip_box(truth.bbox, width, height) if clip else truth.bbox
        for truth in truths
    ]
    prediction_matches: dict[int, int] = {}
    truth_matches: dict[int, int] = {}
    if prediction_order:
        prediction_indices = sorted(range(len(predictions)), key=lambda index: -predictions[index][1])
        for prediction_index in prediction_indices:
            available = [
                (iou(predictions[prediction_index][0], truth_box), truth_index)
                for truth_index, truth_box in enumerate(truth_boxes)
                if truth_index not in truth_matches
            ]
            if not available:
                continue
            score, truth_index = max(available)
            if score >= 0.5:
                prediction_matches[prediction_index] = truth_index
                truth_matches[truth_index] = prediction_index
    else:
        used_predictions: set[int] = set()
        for truth_index, truth_box in enumerate(truth_boxes):
            available = [
                (iou(truth_box, predictions[prediction_index][0]), prediction_index)
                for prediction_index in range(len(predictions))
                if prediction_index not in used_predictions
            ]
            if not available:
                continue
            score, prediction_index = max(available)
            if score >= 0.5:
                prediction_matches[prediction_index] = truth_index
                truth_matches[truth_index] = prediction_index
                used_predictions.add(prediction_index)
    return prediction_matches, truth_matches


def _distance(prediction: tuple[float, float, float, float]) -> float | None:
    height = prediction[3] - prediction[1]
    return FX * PERSON_HEIGHT_M / height if height > 0 else None


def _predicted_lateral(prediction: tuple[float, float, float, float]) -> float | None:
    distance = _distance(prediction)
    if distance is None:
        return None
    centre_x = (prediction[0] + prediction[2]) / 2.0
    return (centre_x - CX) * distance / FX


def _pred_in_corridor(prediction: tuple[float, float, float, float]) -> bool:
    distance = _distance(prediction)
    lateral = _predicted_lateral(prediction)
    return distance is not None and lateral is not None and distance <= FORWARD_LIMIT_M and abs(lateral) <= PRIMARY_CORRIDOR_M


def _critical(truth: Truth) -> bool:
    return classify_zone(truth.forward_m).zone == "critical_danger"


def _predicted_critical(prediction: tuple[float, float, float, float]) -> bool:
    distance = _distance(prediction)
    return distance is not None and classify_zone(distance).zone == "critical_danger"


def _view(
    frames: Sequence[Frame],
    corridor: bool,
    clip: bool = False,
    predicted_corridor: bool = False,
    prediction_order: bool = True,
) -> dict[str, object]:
    total_critical = detected_critical = zone_correct = 0
    confusion: Counter[str] = Counter()
    matched_pairs = 0
    predicted_count = 0
    false_positives = 0
    errors_border: list[float] = []
    errors_nonborder: list[float] = []
    for frame in frames:
        truths = tuple(truth for truth in frame.truths if not corridor or in_corridor(truth, PRIMARY_CORRIDOR_M))
        predictions = tuple(
            prediction for prediction in frame.predictions
            if not predicted_corridor or _pred_in_corridor(prediction[0])
        )
        ignored_truths = tuple(
            truth for truth in frame.truths
            if corridor and not in_corridor(truth, PRIMARY_CORRIDOR_M)
        )
        included_prediction_matches, truth_matches = _match(
            truths, predictions, clip, frame.width, frame.height, prediction_order
        )
        remaining_predictions = [
            prediction for index, prediction in enumerate(predictions)
            if index not in included_prediction_matches
        ]
        ignored_matches, _ = _match(ignored_truths, remaining_predictions, clip, frame.width, frame.height, prediction_order)
        predicted_count += len(predictions)
        false_positives += len(predictions) - len(included_prediction_matches) - len(ignored_matches)
        critical_indices = {index for index, truth in enumerate(truths) if _critical(truth)}
        total_critical += len(critical_indices)
        detected_critical += len(critical_indices.intersection(truth_matches))
        for truth_index, prediction_index in truth_matches.items():
            truth = truths[truth_index]
            prediction = predictions[prediction_index][0]
            truth_zone = classify_zone(truth.forward_m).zone
            predicted_distance = _distance(prediction)
            predicted_zone = classify_zone(predicted_distance).zone if predicted_distance is not None else "invalid"
            confusion[f"{truth_zone}->{predicted_zone}"] += 1
            matched_pairs += 1
            if truth_index in critical_indices and predicted_zone == "critical_danger":
                zone_correct += 1
            if predicted_distance is not None:
                error = abs(predicted_distance - truth.forward_m)
                clipped = _clip_box(truth.bbox, frame.width, frame.height)
                touches_border = (
                    clipped[0] <= 0 or clipped[1] <= 0
                    or clipped[2] >= frame.width or clipped[3] >= frame.height
                )
                (errors_border if touches_border else errors_nonborder).append(error)
    return {
        "ground_truth_critical": total_critical,
        "critical_detected_iou_0.5": detected_critical,
        "detection_fraction": detected_critical / total_critical if total_critical else 0.0,
        "critical_zone_correct": zone_correct,
        "zone_correct_rate_among_detected": zone_correct / detected_critical if detected_critical else 0.0,
        "critical_recall": zone_correct / total_critical if total_critical else 0.0,
        "matched_pairs": matched_pairs,
        "predictions_in_view": predicted_count,
        "false_positives_in_view": false_positives,
        "precision": matched_pairs / (matched_pairs + false_positives) if matched_pairs + false_positives else 0.0,
        "zone_confusion_detected_pairs": dict(sorted(confusion.items())),
        "distance_error_border_mae_m": float(np.mean(errors_border)) if errors_border else None,
        "distance_error_nonborder_mae_m": float(np.mean(errors_nonborder)) if errors_nonborder else None,
        "border_matched_count": len(errors_border),
        "nonborder_matched_count": len(errors_nonborder),
    }


def _strata(frames: Sequence[Frame], corridor: bool) -> list[dict[str, object]]:
    definitions = (
        ("truncation", lambda truth: "0" if truth.truncation == 0 else "0<level<=0.15" if truth.truncation <= 0.15 else ">0.15"),
        ("occlusion", lambda truth: str(truth.occlusion) if truth.occlusion <= 2 else "3+"),
        ("box_height_px", lambda truth: "<25" if truth.bbox[3] - truth.bbox[1] < 25 else "25-40" if truth.bbox[3] - truth.bbox[1] < 40 else ">=40"),
    )
    rows: list[dict[str, object]] = []
    for dimension, labeler in definitions:
        groups: dict[str, list[tuple[Frame, int]]] = defaultdict(list)
        for frame in frames:
            truths = tuple(truth for truth in frame.truths if not corridor or in_corridor(truth, PRIMARY_CORRIDOR_M))
            _, truth_matches = _match(truths, frame.predictions, width=frame.width, height=frame.height)
            for index, truth in enumerate(truths):
                if _critical(truth):
                    groups[labeler(truth)].append((frame, index))
        for label, members in sorted(groups.items()):
            detected = 0
            correct = 0
            for frame, index in members:
                truths = tuple(truth for truth in frame.truths if not corridor or in_corridor(truth, PRIMARY_CORRIDOR_M))
                _, matches = _match(truths, frame.predictions, width=frame.width, height=frame.height)
                if index in matches:
                    detected += 1
                    if _predicted_critical(frame.predictions[matches[index]][0]):
                        correct += 1
            rows.append({
                "view": "primary_corridor" if corridor else "all_pedestrians",
                "dimension": dimension,
                "level": label,
                "ground_truth_critical": len(members),
                "detected_iou_0.5": detected,
                "recall": detected / len(members) if members else 0.0,
                "zone_correct_among_detected": correct / detected if detected else 0.0,
            })
    return rows


def _border_stats(frames: Sequence[Frame]) -> dict[str, object]:
    counts = Counter()
    overflow = Counter()
    for frame in frames:
        for truth in frame.truths:
            x1, y1, x2, y2 = truth.bbox
            amounts = {
                "left": max(0.0, -x1),
                "top": max(0.0, -y1),
                "right": max(0.0, x2 - frame.width),
                "bottom": max(0.0, y2 - frame.height),
            }
            if any(amount > 0 for amount in amounts.values()):
                counts["boxes_outside"] += 1
                for side, amount in amounts.items():
                    if amount > 0:
                        overflow[f"{side}_boxes"] += 1
                        overflow[f"{side}_pixels_total"] += amount
                        overflow[f"{side}_pixels_max"] = max(overflow[f"{side}_pixels_max"], amount)
    return {"boxes_outside_image": counts["boxes_outside"], "overflow_by_side": dict(overflow)}


def _save_worst(frames: Sequence[Frame], output_dir: Path, limit: int = 30) -> None:
    candidates: list[tuple[float, Frame, int, int | None]] = []
    for frame in frames:
        _, matches = _match(frame.truths, frame.predictions, width=frame.width, height=frame.height)
        for truth_index, truth in enumerate(frame.truths):
            if not _critical(truth) or truth_index in matches:
                continue
            best_iou = max((iou(truth.bbox, prediction[0]) for prediction in frame.predictions), default=0.0)
            candidates.append((best_iou, frame, truth_index, None))
    selected: list[tuple[Frame, list[int]]] = []
    for _, frame, truth_index, _ in sorted(candidates, key=lambda item: (item[0], item[1].image_path.name)):
        existing = next((item for item in selected if item[0] == frame), None)
        if existing:
            existing[1].append(truth_index)
        elif len(selected) < limit:
            selected.append((frame, [truth_index]))
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, (frame, missed_indices) in enumerate(selected, 1):
        image = cv2.imread(str(frame.image_path))
        if image is None:
            continue
        for truth_index, truth in enumerate(frame.truths):
            color = (0, 0, 255)
            thickness = 3 if truth_index in missed_indices else 2
            cv2.rectangle(image, (int(truth.bbox[0]), int(truth.bbox[1])), (int(truth.bbox[2]), int(truth.bbox[3])), color, thickness)
        for prediction, confidence in frame.predictions:
            cv2.rectangle(image, (int(prediction[0]), int(prediction[1])), (int(prediction[2]), int(prediction[3])), (0, 255, 0), 2)
            cv2.putText(image, f"{confidence:.2f}", (int(prediction[0]), max(15, int(prediction[1]) - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.imwrite(str(output_dir / f"worst_missed_critical_{index:02d}_{frame.image_path.stem}.jpg"), image)


def _write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(weights: Path, images_dir: Path, labels_dir: Path, output_dir: Path, imgsz: int, confidence: float) -> dict[str, object]:
    """Run the isolated critical-recall diagnosis."""
    model = YOLO(str(weights))
    frames: list[Frame] = []
    images = sorted(path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    for index, image_path in enumerate(images, 1):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        result = model.predict(source=str(image_path), imgsz=imgsz, conf=confidence, verbose=False)[0]
        predictions = tuple(
            (tuple(float(value) for value in box.xyxy[0].tolist()), float(box.conf[0]))
            for box in result.boxes if int(box.cls[0]) == 0
        )
        frames.append(Frame(image_path, width, height, _parse_truth(labels_dir / f"{image_path.stem}.txt"), predictions))
        if index % 500 == 0 or index == len(images):
            LOGGER.info("Diagnosed %d/%d images", index, len(images))

    all_view = _view(frames, corridor=False)
    corridor_view = _view(frames, corridor=True)
    corridor_predicted_view = _view(frames, corridor=True, predicted_corridor=True)
    clipped_all = _view(frames, corridor=False, clip=True)
    clipped_corridor = _view(frames, corridor=True, clip=True)
    v1_like = _view(frames, corridor=False, prediction_order=False)
    report: dict[str, object] = {
        "matching_setup": {
            "confidence_threshold": confidence,
            "imgsz": imgsz,
            "class_filter": "class_id == 0",
            "iou_threshold": 0.5,
            "matching_v2": "confidence-ordered one-to-one",
            "matching_v1": "ground-truth-ordered one-to-one",
            "geometric_depth": "Z = 621 * 1.7 / predicted box height",
            "critical_threshold_m": 7.0,
            "primary_corridor": "|GT lateral| <= 2.0 m and GT forward <= 30 m",
            "predicted_corridor": "|(u-cx)*Z/fx| <= 2.0 m and Z <= 30 m",
        },
        "v1_explanation": {
            "formula": "matched critical GT predicted critical / matched critical GT",
            "why_90_71_percent": "The original v1 zone_rows contained only IoU-matched objects, so missed critical pedestrians were absent from the denominator.",
            "v1_like_recomputed": v1_like,
        },
        "decomposition": {
            "all_pedestrians": all_view,
            "primary_corridor_ground_truth": corridor_view,
            "primary_corridor_ground_truth_and_prediction": corridor_predicted_view,
        },
        "stratified_critical_ground_truth": _strata(frames, False) + _strata(frames, True),
        "image_boundary": _border_stats(frames),
        "clipped_matching_change": {
            "all_pedestrians_unclipped": all_view,
            "all_pedestrians_clipped": clipped_all,
            "primary_corridor_unclipped": corridor_view,
            "primary_corridor_clipped": clipped_corridor,
        },
        "v1_v2_difference": {
            "explanation": "v1 matched each ground-truth box to its best unused prediction in GT order; v2 matches predictions in descending confidence order. The three-object count difference comes from these competing one-to-one assignments, not from changed weights or thresholds.",
            "v1_counts": {"tp": 5538, "fp": 1332, "fn": 4035, "precision": 0.8061},
            "v2_counts": {"tp": 5535, "fp": 1335, "fn": 4038, "precision": 0.8056768558951966},
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "critical_diagnosis.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _write_csv(output_dir / "critical_strata.csv", report["stratified_critical_ground_truth"])
    _save_worst(frames, output_dir / "worst_missed_critical")
    return report


def main() -> None:
    """Run the diagnosis and print findings before implementation notes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=Path("yolo11n.pt"))
    parser.add_argument("--images", type=Path, default=Path("dataset/images"))
    parser.add_argument("--labels", type=Path, default=Path("dataset/labels"))
    parser.add_argument("--output", type=Path, default=Path("outputs/results_v2/diagnosis"))
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    report = evaluate(args.weights, args.images, args.labels, args.output, args.imgsz, args.conf)
    print("FINDINGS")
    print(json.dumps({
        "v1_explanation": report["v1_explanation"],
        "decomposition": report["decomposition"],
        "image_boundary": report["image_boundary"],
        "clipped_matching_change": report["clipped_matching_change"],
        "v1_v2_difference": report["v1_v2_difference"],
    }, indent=2))
    print("\nBUG FIXES")
    print("No v1/v2 evaluator or result artifact was modified; this diagnosis is isolated.")


if __name__ == "__main__":
    main()

"""Evaluation metrics and result exports for Guardian Vision."""

from __future__ import annotations

import csv
import time
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np

from src.risk import ZoneThresholds, classify_zone

DISTANCE_BINS = ((0.0, 10.0, "0-10"), (10.0, 20.0, "10-20"), (20.0, 30.0, "20-30"), (30.0, float("inf"), "30+"))


def distance_metrics(rows: Iterable[Mapping[str, float]]) -> list[dict[str, float | str]]:
    """Compute MAE and AbsRel per ground-truth distance bin."""
    grouped: dict[str, list[tuple[float, float]]] = {name: [] for _, _, name in DISTANCE_BINS}
    for row in rows:
        truth, estimate = float(row["ground_truth_distance"]), float(row["estimated_distance"])
        name = next(label for low, high, label in DISTANCE_BINS if low <= truth < high)
        grouped[name].append((truth, estimate))
    result = []
    for name, values in grouped.items():
        if not values:
            result.append({"distance_bin_m": name, "count": 0, "MAE_m": float("nan"), "AbsRel": float("nan")})
            continue
        truth = np.asarray([item[0] for item in values])
        estimate = np.asarray([item[1] for item in values])
        result.append({"distance_bin_m": name, "count": len(values), "MAE_m": float(np.mean(np.abs(estimate - truth))), "AbsRel": float(np.mean(np.abs(estimate - truth) / np.maximum(truth, 1e-8)))})
    return result


def zone_confusion(rows: Iterable[Mapping[str, float]], thresholds: ZoneThresholds) -> dict[tuple[str, str], int]:
    """Return a ground-truth/predicted zone confusion counter."""
    matrix: Counter[tuple[str, str]] = Counter()
    for row in rows:
        truth = classify_zone(float(row["ground_truth_distance"]), thresholds).zone
        predicted = classify_zone(float(row["estimated_distance"]), thresholds).zone
        matrix[(truth, predicted)] += 1
    return dict(matrix)


def critical_recall_false_brake(rows: Iterable[Mapping[str, float]], thresholds: ZoneThresholds) -> tuple[float, float]:
    """Return critical-zone recall and false-brake rate."""
    data = list(rows)
    truth_critical = [row for row in data if classify_zone(float(row["ground_truth_distance"]), thresholds).zone == "critical_danger"]
    predicted_critical = [row for row in data if classify_zone(float(row["estimated_distance"]), thresholds).zone == "critical_danger"]
    recall = sum(classify_zone(float(row["estimated_distance"]), thresholds).zone == "critical_danger" for row in truth_critical) / len(truth_critical) if truth_critical else float("nan")
    noncritical = [row for row in data if classify_zone(float(row["ground_truth_distance"]), thresholds).zone != "critical_danger"]
    false_brake = sum(classify_zone(float(row["estimated_distance"]), thresholds).zone == "critical_danger" for row in noncritical) / len(noncritical) if noncritical else float("nan")
    return float(recall), float(false_brake)


def threshold_ablation(rows: Iterable[Mapping[str, float]], critical_values: Iterable[float], warning_values: Iterable[float]) -> list[dict[str, float]]:
    """Sweep critical/warning boundaries and report safety trade-offs."""
    data = list(rows)
    result = []
    for critical in critical_values:
        for warning in warning_values:
            if warning <= critical:
                continue
            recall, false_brake = critical_recall_false_brake(data, ZoneThresholds(critical, warning))
            result.append({"critical_m": critical, "warning_m": warning, "critical_recall": recall, "false_brake_rate": false_brake})
    return result


def evaluate_timed(estimator, samples: Iterable[Mapping[str, object]]) -> tuple[list[dict[str, float]], float]:
    """Evaluate an estimator while measuring wall-clock inference time."""
    rows = []
    durations = []
    for sample in samples:
        start = time.perf_counter()
        estimated = float(estimator(sample["image"], sample["bbox"]))
        durations.append(time.perf_counter() - start)
        rows.append({"ground_truth_distance": float(sample["ground_truth_distance"]), "estimated_distance": estimated})
    return rows, float(np.mean(durations)) if durations else float("nan")


def export_results(results_dir: Path, model_name: str, distance_rows: list[dict], ablation_rows: list[dict]) -> None:
    """Export metric tables and a threshold trade-off plot."""
    results_dir.mkdir(parents=True, exist_ok=True)
    for filename, rows in ((f"{model_name}_distance.csv", distance_rows), (f"{model_name}_threshold_ablation.csv", ablation_rows)):
        if not rows:
            continue
        with (results_dir / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    if ablation_rows:
        plt.figure()
        plt.scatter([row["false_brake_rate"] for row in ablation_rows], [row["critical_recall"] for row in ablation_rows])
        plt.xlabel("False-brake rate")
        plt.ylabel("Critical-zone recall")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(results_dir / f"{model_name}_threshold_tradeoff.png", dpi=150)
        plt.close()

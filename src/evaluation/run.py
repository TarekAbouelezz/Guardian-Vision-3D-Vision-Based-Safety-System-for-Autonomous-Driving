"""Export Guardian Vision metrics from a prediction CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.evaluation.metrics import (
    critical_recall_false_brake,
    distance_metrics,
    export_results,
    threshold_ablation,
    zone_confusion,
)
from src.risk import ZoneThresholds


def main() -> None:
    """Read rows and write metric tables/plots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path, help="CSV with estimator, ground_truth_distance, estimated_distance")
    parser.add_argument("--output", type=Path, default=Path("outputs/results"))
    parser.add_argument("--critical", type=float, default=7.0)
    parser.add_argument("--warning", type=float, default=15.0)
    args = parser.parse_args()
    with args.predictions.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    thresholds = ZoneThresholds(args.critical, args.warning)
    for estimator in sorted({row.get("estimator", "unknown") for row in rows}):
        selected = [row for row in rows if row.get("estimator", "unknown") == estimator]
        distance_rows = distance_metrics(selected)
        recall, false_brake = critical_recall_false_brake(selected, thresholds)
        distance_rows.append({"distance_bin_m": "overall", "count": len(selected), "MAE_m": sum(float(r["MAE_m"]) * int(r["count"]) for r in distance_rows if int(r["count"])) / max(len(selected), 1), "AbsRel": sum(float(r["AbsRel"]) * int(r["count"]) for r in distance_rows if int(r["count"])) / max(len(selected), 1), "critical_recall": recall, "false_brake_rate": false_brake, "zone_confusion": zone_confusion(selected, thresholds)})
        export_results(args.output, estimator, distance_rows, threshold_ablation(selected, range(3, 16), range(8, 26, 2)))


if __name__ == "__main__":
    main()

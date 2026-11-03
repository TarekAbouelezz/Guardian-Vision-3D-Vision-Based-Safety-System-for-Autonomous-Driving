"""Fine-tune YOLO and evaluate pedestrian recall/mAP50 by metric distance."""

from __future__ import annotations

import argparse
import csv
import logging
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from ultralytics import YOLO

LOGGER = logging.getLogger(__name__)
DISTANCE_BINS = ((0.0, 10.0, "0-10"), (10.0, 20.0, "10-20"), (20.0, 30.0, "20-30"), (30.0, float("inf"), "30+"))


def parse_pedestrians(label_path: Path) -> list[tuple[float, float, float, float, float]]:
    """Read KITTI pedestrian boxes and metric Z values."""
    pedestrians = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) >= 14 and fields[0].lower() == "pedestrian":
            pedestrians.append(tuple(float(value) for value in (fields[4], fields[5], fields[6], fields[7], fields[13])))
    return pedestrians


def iou(left: Iterable[float], right: Iterable[float]) -> float:
    """Calculate intersection over union for two ``xyxy`` boxes."""
    ax1, ay1, ax2, ay2 = left
    bx1, by1, bx2, by2 = right
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def _average_precision(scores: list[float], matches: list[int], positives: int) -> float:
    if positives == 0:
        return float("nan")
    order = np.argsort(-np.asarray(scores))
    tp = np.cumsum(np.asarray(matches)[order])
    fp = np.cumsum(1 - np.asarray(matches)[order])
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / positives
    return float(np.max([precision[recall >= level].max() if np.any(recall >= level) else 0.0 for level in np.linspace(0, 1, 101)]))


def evaluate_model(model_path: Path, image_dir: Path, test_ids: Path, kitti_labels: Path, output_csv: Path, imgsz: int = 1280) -> list[dict[str, object]]:
    """Evaluate one model and export recall/AP50 grouped by ground-truth distance."""
    model = YOLO(str(model_path))
    records: dict[str, dict[str, list[float] | int]] = defaultdict(lambda: {"scores": [], "matches": [], "positives": 0})
    for manifest_entry in (line.strip() for line in test_ids.read_text(encoding="utf-8").splitlines() if line.strip()):
        frame_id = Path(manifest_entry).stem
        image_candidates = list(image_dir.glob(f"{frame_id}.*"))
        if not image_candidates:
            LOGGER.warning("Skipping missing image %s", frame_id)
            continue
        ground_truth = parse_pedestrians(kitti_labels / f"{frame_id}.txt")
        prediction = model.predict(source=str(image_candidates[0]), imgsz=imgsz, verbose=False)[0]
        predicted = [
            (box.xyxy[0].tolist(), float(box.conf[0]))
            for box in prediction.boxes
            if int(box.cls[0]) == 0
        ]
        used: set[int] = set()
        for gt_box in ground_truth:
            bin_name = next(name for low, high, name in DISTANCE_BINS if low <= gt_box[4] < high)
            records[bin_name]["positives"] += 1
            best = max(
                ((iou(gt_box[:4], candidate[0]), index) for index, candidate in enumerate(predicted) if index not in used),
                default=(0.0, -1),
            )
            best_score = predicted[best[1]][1] if best[1] >= 0 else 0.0
            records[bin_name]["scores"].append(best_score)
            if best[0] >= 0.5:
                used.add(best[1])
                records[bin_name]["matches"].append(1)
            else:
                records[bin_name]["matches"].append(0)

    rows = []
    for _, _, bin_name in DISTANCE_BINS:
        data = records[bin_name]
        positives = int(data["positives"])
        matches = int(sum(data["matches"]))
        rows.append({"model": model_path.stem, "distance_bin_m": bin_name, "recall": matches / positives if positives else float("nan"), "mAP50": _average_precision(data["scores"], data["matches"], positives), "ground_truth": positives})
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return rows


def train(weights: Path, data_yaml: Path, output_dir: Path, epochs: int, imgsz: int, device: str) -> Path:
    """Fine-tune YOLO11n and return the best checkpoint."""
    model = YOLO(str(weights))
    result = model.train(data=str(data_yaml), epochs=epochs, imgsz=imgsz, device=device, project=str(output_dir), name="guardian_yolo11n")
    best = Path(result.save_dir) / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"Training completed without a best checkpoint: {best}")
    return best


def main() -> None:
    """Run training and/or distance-binned evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=Path("yolo11n.pt"))
    parser.add_argument("--data", type=Path, default=Path("dataset/yolo/carla.yaml"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("outputs/results/distance_metrics.csv"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    model_path = train(args.weights, args.data, Path("outputs/training"), args.epochs, args.imgsz, args.device) if args.train else args.weights
    if args.evaluate:
        rows = evaluate_model(model_path, Path("dataset/images"), Path("dataset/yolo/test.txt"), Path("dataset/labels"), args.output, args.imgsz)
        for row in rows:
            print(f"{row['model']}: {row['distance_bin_m']} m recall={row['recall']:.3f} mAP50={row['mAP50']:.3f}")


if __name__ == "__main__":
    main()

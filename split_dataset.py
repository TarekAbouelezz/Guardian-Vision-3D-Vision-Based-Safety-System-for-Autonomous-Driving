"""Create sequence-disjoint YOLO train/validation/test splits for CARLA."""

from __future__ import annotations

import argparse
import csv
import logging
import os
import shutil
from collections import defaultdict
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def _allocate_groups(groups: list[tuple[str, list[str]]], ratios: tuple[float, float, float]) -> dict[str, list[str]]:
    """Greedily allocate whole sequences while matching target frame counts."""
    total = sum(len(frames) for _, frames in groups)
    targets = [total * ratio for ratio in ratios]
    result: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    counts = [0, 0, 0]
    for group_id, frames in sorted(groups, key=lambda item: (-len(item[1]), item[0])):
        index = min(range(3), key=lambda candidate: counts[candidate] - targets[candidate])
        result[("train", "val", "test")[index]].extend(frames)
        counts[index] += len(frames)
    return result


def _link_or_copy(source: Path, destination: Path) -> None:
    """Create a hard link when possible, otherwise copy the image."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def split_dataset(
    dataset_dir: Path,
    output_dir: Path,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> dict[str, list[str]]:
    """Split by ``run_id`` (sequence), retaining scenario boundaries."""
    metadata_path = dataset_dir / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")
    with metadata_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("metadata.csv contains no frames")

    groups: defaultdict[str, list[str]] = defaultdict(list)
    for row in rows:
        frame_id = f"{int(row['frame_id']):06d}"
        group_id = f"{row['run_id']}::{row['scenario']}"
        groups[group_id].append(frame_id)

    splits = _allocate_groups(list(groups.items()), ratios)
    for split_name, frame_ids in splits.items():
        split_root = output_dir / split_name
        image_paths: list[str] = []
        for frame_id in sorted(frame_ids):
            image_candidates = list((dataset_dir / "images").glob(f"{frame_id}.*"))
            if not image_candidates:
                raise FileNotFoundError(f"No image found for frame {frame_id}")
            image_path = split_root / "images" / image_candidates[0].name
            _link_or_copy(image_candidates[0], image_path)
            image_paths.append(str(image_path.resolve()))
            source_label = dataset_dir / "yolo" / "labels" / f"{frame_id}.txt"
            if not source_label.exists():
                raise FileNotFoundError(f"Missing converted label: {source_label}")
            label_destination = split_root / "labels" / source_label.name
            label_destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_label, label_destination)
        (output_dir / f"{split_name}.txt").write_text(
            "\n".join(image_paths) + "\n",
            encoding="utf-8",
        )

    yaml_path = output_dir / "carla.yaml"
    yaml_path.write_text(
        "path: .\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n"
        "names:\n"
        "  0: Pedestrian\n",
        encoding="utf-8",
    )
    LOGGER.info("Wrote sequence-disjoint splits: %s", {name: len(items) for name, items in splits.items()})
    return splits


def main() -> None:
    """Run the sequence split command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, default=Path("dataset/yolo"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    split_dataset(args.dataset, args.output)


if __name__ == "__main__":
    main()

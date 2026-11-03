"""Convert the CARLA KITTI-style annotations into YOLO detection labels."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

LOGGER = logging.getLogger(__name__)
IMAGE_WIDTH = 1242.0
IMAGE_HEIGHT = 375.0


def convert_line(line: str, image_width: float = IMAGE_WIDTH, image_height: float = IMAGE_HEIGHT) -> str | None:
    """Convert one KITTI object row to ``class x_center y_center width height``."""
    fields = line.split()
    if len(fields) < 8 or fields[0].lower() != "pedestrian":
        return None

    try:
        x1, y1, x2, y2 = (float(value) for value in fields[4:8])
    except ValueError as exc:
        raise ValueError(f"Invalid bounding box in annotation: {line.rstrip()}") from exc

    width = x2 - x1
    height = y2 - y1
    if width <= 0 or height <= 0:
        raise ValueError(f"Non-positive bounding box in annotation: {line.rstrip()}")

    x_center = (x1 + x2) / 2.0 / image_width
    y_center = (y1 + y2) / 2.0 / image_height
    normalized_width = width / image_width
    normalized_height = height / image_height
    values = (0, x_center, y_center, normalized_width, normalized_height)
    return " ".join(f"{value:.8f}" if index else str(value) for index, value in enumerate(values))


def convert_labels(
    source_dir: Path,
    output_dir: Path,
    image_width: float = IMAGE_WIDTH,
    image_height: float = IMAGE_HEIGHT,
) -> tuple[int, int]:
    """Convert all source annotations and return ``(files, pedestrian_boxes)``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    file_count = box_count = 0
    for source_path in sorted(source_dir.glob("*.txt")):
        converted: list[str] = []
        for line in source_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            result = convert_line(line, image_width, image_height)
            if result is not None:
                converted.append(result)
        (output_dir / source_path.name).write_text(
            "\n".join(converted) + ("\n" if converted else ""),
            encoding="utf-8",
        )
        file_count += 1
        box_count += len(converted)
    LOGGER.info("Converted %d files and %d pedestrian boxes", file_count, box_count)
    return file_count, box_count


def main() -> None:
    """Run the annotation conversion command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("dataset/labels"))
    parser.add_argument("--output", type=Path, default=Path("dataset/yolo/labels"))
    parser.add_argument("--width", type=float, default=IMAGE_WIDTH)
    parser.add_argument("--height", type=float, default=IMAGE_HEIGHT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    convert_labels(args.source, args.output, args.width, args.height)


if __name__ == "__main__":
    main()

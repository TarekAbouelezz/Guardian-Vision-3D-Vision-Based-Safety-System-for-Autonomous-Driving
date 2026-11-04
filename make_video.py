"""Create an MP4 presentation video from annotated Guardian Vision images."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def create_video(image_dir: Path, output_path: Path, fps: float = 20.0) -> int:
    """Write sorted annotated images to an MP4 and return the frame count."""
    images = sorted(image_dir.glob("*_guardian.jpg"))
    if not images:
        raise FileNotFoundError(f"No annotated images found in {image_dir}")
    first = cv2.imread(str(images[0]))
    if first is None:
        raise RuntimeError(f"Unable to read first image: {images[0]}")
    height, width = first.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Unable to open video writer: {output_path}")
    try:
        for image_path in images:
            frame = cv2.imread(str(image_path))
            if frame is None:
                raise RuntimeError(f"Unable to read annotated image: {image_path}")
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
            writer.write(frame)
    finally:
        writer.release()
    return len(images)


def main() -> None:
    """Create a video from a directory of annotated frames."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("outputs/fulltest_6566"))
    parser.add_argument("--output", type=Path, default=Path("outputs/fulltest_6566_guardian.mp4"))
    parser.add_argument("--fps", type=float, default=20.0)
    args = parser.parse_args()
    count = create_video(args.input, args.output, args.fps)
    print(f"Created {args.output} with {count} frames at {args.fps:g} FPS")


if __name__ == "__main__":
    main()

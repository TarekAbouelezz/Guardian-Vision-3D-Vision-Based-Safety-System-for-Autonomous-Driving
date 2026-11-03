from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Sequence

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results

from src.detection.base import BaseDetector, Detection
from src.visualization.visualizer import Visualizer


class YOLODetector(BaseDetector):
    """Pedestrian detector built on top of an Ultralytics YOLO model."""

    PERSON_CLASS_NAME = "person"

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        confidence_threshold: float = 0.0,
        device: str = "cpu",
        output_images_dir: str = "outputs/images",
        output_videos_dir: str = "outputs/videos",
    ) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.output_videos_dir = Path(output_videos_dir)
        self.visualizer = Visualizer(image_output_dir=Path(output_images_dir))
        self.model = self._load_model(model_path)

    def _load_model(self, model_path: str) -> YOLO:
        """Load the configured YOLO model with error handling."""
        try:
            model = YOLO(model_path)
            self.logger.info("Loaded YOLO model from %s", model_path)
            return model
        except Exception as exc:
            self.logger.exception("Could not load YOLO model: %s", model_path)
            raise RuntimeError(f"Could not load YOLO model from: {model_path}") from exc

    def detect_image(self, image_path: str) -> List[Detection]:
        """Run person detection on a single image file."""
        image_file = Path(image_path)
        if not image_file.exists():
            raise FileNotFoundError(f"Image not found: {image_file}")

        results = self.model.predict(source=str(image_file), verbose=False, device=self.device)
        detections = self._extract_person_detections(results)

        self.logger.info(
            "Detected %d person(s) in image %s",
            len(detections),
            image_file,
        )
        return detections

    def draw_detections(
        self,
        image: np.ndarray,
        detections: Sequence[Detection],
        output_name: str | None = None,
    ) -> str:
        """Draw person detections and save the annotated image."""
        return self.visualizer.draw_and_save_image(
            image=image,
            detections=detections,
            output_name=output_name,
        )

    def detect_video(self, video_path: str) -> str:
        """Run person detection on a video and save an annotated output video."""
        video_file = Path(video_path)
        if not video_file.exists():
            raise FileNotFoundError(f"Video not found: {video_file}")

        cap = cv2.VideoCapture(str(video_file))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {video_file}")

        output_dir = self.output_videos_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{video_file.stem}_detected.mp4"

        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if fps > 0 else 30.0
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (frame_width, frame_height),
        )

        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Failed to initialize output video writer: {output_path}")

        frame_index = 0
        total_detections = 0

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                detections = self._detect_frame(frame)
                total_detections += len(detections)

                annotated = self.visualizer.draw_on_image(frame, detections)
                writer.write(annotated)

                frame_index += 1
                if frame_index % 50 == 0:
                    self.logger.info(
                        "Processed %d frames from %s",
                        frame_index,
                        video_file.name,
                    )
        finally:
            cap.release()
            writer.release()

        self.logger.info(
            "Video detection finished: frames=%d, total_person_detections=%d, output=%s",
            frame_index,
            total_detections,
            output_path,
        )
        return str(output_path)

    def detect_image_array(self, image: np.ndarray) -> List[Detection]:
        """Run person detection on an in-memory BGR image."""
        if image is None or image.size == 0:
            raise ValueError("image must be a non-empty numpy array")
        return self._detect_frame(image)

    def _detect_frame(self, frame: np.ndarray) -> List[Detection]:
        """Run person detection on a numpy image frame."""
        results = self.model.predict(source=frame, verbose=False, device=self.device)
        return self._extract_person_detections(results)

    def _extract_person_detections(self, results: Sequence[Results]) -> List[Detection]:
        """Convert YOLO predictions into a generic person-only detection schema."""
        detections: List[Detection] = []

        for result in results:
            names = result.names if hasattr(result, "names") else {}
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                class_id = int(box.cls.item())
                class_name = self._resolve_class_name(names, class_id)
                if class_name.lower() != self.PERSON_CLASS_NAME:
                    continue

                confidence = float(box.conf.item())
                if confidence < self.confidence_threshold:
                    continue

                xyxy = box.xyxy[0].tolist()
                bbox = [int(round(value)) for value in xyxy]

                detections.append(
                    Detection(
                        class_id=class_id,
                        class_name=class_name,
                        confidence=confidence,
                        bbox=bbox,
                    )
                )

        detections.sort(key=lambda item: item["confidence"], reverse=True)
        return detections

    @staticmethod
    def _resolve_class_name(names: Dict[int, str] | List[str], class_id: int) -> str:
        """Resolve a class name from either dict or list YOLO class mappings."""
        if isinstance(names, dict):
            return str(names.get(class_id, str(class_id)))

        if isinstance(names, list) and 0 <= class_id < len(names):
            return str(names[class_id])

        return str(class_id)

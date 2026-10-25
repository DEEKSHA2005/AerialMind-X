"""Vision-based object detection for AerialMind-X."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2


@dataclass(frozen=True)
class Detection:
    """Represent one detected object in an image."""

    label: str
    confidence: float
    bounding_box: tuple[int, int, int, int]


class ObstacleDetector:
    """Detect objects in images using a pretrained YOLO model."""

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        confidence_threshold: float = 0.25,
        model: Any | None = None,
    ) -> None:
        """Initialize the detector.

        An injected model can be used for testing without downloading weights.
        """
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("Confidence threshold must be between 0 and 1.")

        self.confidence_threshold = confidence_threshold

        if model is None:
            from ultralytics import YOLO

            model = YOLO(model_path)

        self.model = model

    def detect(self, image_path: str | Path) -> list[Detection]:
        """Detect objects and return labels, confidence, and bounding boxes."""
        image_path = Path(image_path)

        if not image_path.is_file():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        results = self.model.predict(
            source=image,
            conf=self.confidence_threshold,
            verbose=False,
        )

        detections: list[Detection] = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                coordinates = box.xyxy[0].tolist()
                confidence = float(box.conf[0].item())
                class_id = int(box.cls[0].item())

                label = result.names[class_id]

                detections.append(
                    Detection(
                        label=label,
                        confidence=confidence,
                        bounding_box=tuple(
                            round(value) for value in coordinates
                 ),
                    )
                )

        return detections

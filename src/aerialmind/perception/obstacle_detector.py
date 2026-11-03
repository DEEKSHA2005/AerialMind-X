"""Vision-based object detection for AerialMind-X."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    """Represent one detected object in an image."""

    label: str
    confidence: float
    bounding_box: tuple[int, int, int, int]


class ObstacleDetector:
    """Detect objects in images using a YOLO model."""

    def __init__(
        self,
        model_path: str = (
            "artifacts/training/visdrone_full_5epochs/weights/best.pt"
        ),
        confidence_threshold: float = 0.25,
        model: Any | None = None,
    ) -> None:
        """Initialize the detector."""

        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("Confidence threshold must be between 0 and 1.")

        self.confidence_threshold = confidence_threshold

        if model is None:
            from ultralytics import YOLO

            model = YOLO(model_path)

        self.model = model

    def detect(
        self,
        image: str | Path | np.ndarray,
    ) -> list[Detection]:
        """Detect objects from an image path or an OpenCV image array."""

        if isinstance(image, (str, Path)):
            image_path = Path(image)

            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Image file not found: {image_path}"
                )

            image_data = cv2.imread(str(image_path))

            if image_data is None:
                raise ValueError(
                    f"Could not read image: {image_path}"
                )

        elif isinstance(image, np.ndarray):
            if (
                image.size == 0
                or image.ndim != 3
                or image.shape[2] != 3
            ):
                raise ValueError(
                    "Image array must be a non-empty three-channel image."
                )

            image_data = image

        else:
            raise TypeError(
                "Image must be a file path or NumPy image array."
            )

        results = self.model.predict(
            source=image_data,
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

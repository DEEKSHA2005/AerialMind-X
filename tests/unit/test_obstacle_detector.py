from unittest.mock import Mock

import numpy as np
import pytest

from aerialmind.perception.obstacle_detector import ObstacleDetector


def make_fake_model():
    box = Mock()
    box.xyxy = np.array([[10.0, 20.0, 110.0, 120.0]])
    box.conf = np.array([0.91])
    box.cls = np.array([0.0])

    result = Mock()
    result.boxes = [box]
    result.names = {0: "person"}

    model = Mock()
    model.predict.return_value = [result]

    return model


def test_detect_returns_object_information(monkeypatch, tmp_path):
    image_path = tmp_path / "sample.jpg"
    image_path.touch()

    monkeypatch.setattr(
        "aerialmind.perception.obstacle_detector.cv2.imread",
        lambda _: np.zeros((200, 200, 3), dtype=np.uint8),
    )

    model = make_fake_model()
    detector = ObstacleDetector(model=model)

    detections = detector.detect(image_path)

    assert len(detections) == 1
    assert detections[0].label == "person"
    assert detections[0].confidence == pytest.approx(0.91)
    assert detections[0].bounding_box == (10, 20, 110, 120)


def test_rejects_missing_image():
    detector = ObstacleDetector(model=Mock())

    with pytest.raises(FileNotFoundError, match="Image file not found"):
        detector.detect("missing_image.jpg")


def test_rejects_unreadable_image(monkeypatch, tmp_path):
    image_path = tmp_path / "broken.jpg"
    image_path.touch()

    monkeypatch.setattr(
        "aerialmind.perception.obstacle_detector.cv2.imread",
        lambda _: None,
    )

    detector = ObstacleDetector(model=Mock())

    with pytest.raises(ValueError, match="Could not read image"):
        detector.detect(image_path)


def test_rejects_invalid_confidence_threshold():
    with pytest.raises(ValueError, match="between 0 and 1"):
        ObstacleDetector(confidence_threshold=1.5, model=Mock())


def test_returns_empty_list_when_no_objects_detected(
    monkeypatch, tmp_path
):
    image_path = tmp_path / "empty_scene.jpg"
    image_path.touch()

    monkeypatch.setattr(
        "aerialmind.perception.obstacle_detector.cv2.imread",
        lambda _: np.zeros((200, 200, 3), dtype=np.uint8),
    )

    result = Mock()
    result.boxes = []

    model = Mock()
    model.predict.return_value = [result]

    detector = ObstacleDetector(model=model)

    assert detector.detect(image_path) == []

import pytest

from aerialmind.perception.obstacle_detector import Detection
from aerialmind.perception.vision_grid_mapper import (
    detections_to_grid_obstacles,
)


def test_maps_detection_center_to_grid_cell():
    detections = [
        Detection("person", 0.95, (40, 20, 60, 40)),
    ]

    obstacles = detections_to_grid_obstacles(
        detections,
        image_width=100,
        image_height=100,
        grid_rows=5,
        grid_cols=5,
    )

    assert obstacles == {(1, 2)}


def test_ignores_low_confidence_detections():
    detections = [
        Detection("person", 0.20, (40, 20, 60, 40)),
    ]

    obstacles = detections_to_grid_obstacles(
        detections, 100, 100, 5, 5, confidence_threshold=0.25
    )

    assert obstacles == set()


def test_multiple_detections_in_same_cell_are_deduplicated():
    detections = [
        Detection("person", 0.90, (40, 20, 50, 30)),
        Detection("car", 0.85, (45, 25, 55, 35)),
    ]

    obstacles = detections_to_grid_obstacles(
        detections, 100, 100, 5, 5
    )

    assert obstacles == {(1, 2)}


def test_ignores_detection_center_outside_image():
    detections = [
        Detection("person", 0.90, (-30, 20, -10, 40)),
    ]

    obstacles = detections_to_grid_obstacles(
        detections, 100, 100, 5, 5
    )

    assert obstacles == set()


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ((0, 100, 5, 5), "Image width"),
        ((100, 0, 5, 5), "Image height"),
        ((100, 100, 0, 5), "Grid rows"),
        ((100, 100, 5, 0), "Grid columns"),
    ],
)
def test_rejects_invalid_dimensions(arguments, message):
    with pytest.raises(ValueError, match=message):
        detections_to_grid_obstacles(
            [],
            image_width=arguments[0],
            image_height=arguments[1],
            grid_rows=arguments[2],
            grid_cols=arguments[3],
        )


def test_rejects_invalid_confidence_threshold():
    with pytest.raises(ValueError, match="between 0 and 1"):
        detections_to_grid_obstacles(
            [], 100, 100, 5, 5, confidence_threshold=1.5
        )


def test_rejects_invalid_bounding_box():
    detections = [
        Detection("person", 0.90, (60, 20, 40, 40)),
    ]

    with pytest.raises(ValueError, match="bounding box"):
        detections_to_grid_obstacles(
            detections, 100, 100, 5, 5
        )

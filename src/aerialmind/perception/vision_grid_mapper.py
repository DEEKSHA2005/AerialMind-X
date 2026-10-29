"""Map image detections to occupancy-grid obstacle cells."""

from __future__ import annotations

from collections.abc import Iterable

from aerialmind.perception.obstacle_detector import Detection
from aerialmind.planning.astar import Coordinate


def detections_to_grid_obstacles(
    detections: Iterable[Detection],
    image_width: int,
    image_height: int,
    grid_rows: int,
    grid_cols: int,
    confidence_threshold: float = 0.25,
) -> set[Coordinate]:
    """Convert detection bounding-box centers into simulated grid cells.

    The mapping assumes the image represents the complete simulated grid
    area. It is an approximation for simulation, not real-world localization.
    """
    if type(image_width) is not int or image_width <= 0:
        raise ValueError("Image width must be a positive integer.")

    if type(image_height) is not int or image_height <= 0:
        raise ValueError("Image height must be a positive integer.")

    if type(grid_rows) is not int or grid_rows <= 0:
        raise ValueError("Grid rows must be a positive integer.")

    if type(grid_cols) is not int or grid_cols <= 0:
        raise ValueError("Grid columns must be a positive integer.")

    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("Confidence threshold must be between 0 and 1.")

    obstacles: set[Coordinate] = set()

    for detection in detections:
        if detection.confidence < confidence_threshold:
            continue

        x1, y1, x2, y2 = detection.bounding_box

        if x1 > x2 or y1 > y2:
            raise ValueError("Detection bounding box coordinates are invalid.")

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # Ignore detections whose centers fall outside the image.
        if not (0 <= center_x < image_width):
            continue

        if not (0 <= center_y < image_height):
            continue

        col = int(center_x * grid_cols / image_width)
        row = int(center_y * grid_rows / image_height)

        obstacles.add((row, col))

    return obstacles

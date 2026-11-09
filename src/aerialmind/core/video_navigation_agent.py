"""Sequential video-based autonomous navigation for AerialMind-X."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from aerialmind.control.drone_simulator import DroneSimulator
from aerialmind.perception.obstacle_detector import (
    Detection,
    ObstacleDetector,
)
from aerialmind.perception.vision_grid_mapper import (
    detections_to_grid_obstacles,
)
from aerialmind.planning.astar import Coordinate, astar


@dataclass
class VideoFrameResult:
    """Store navigation results for one processed video frame."""

    frame_index: int
    detections: list[Detection]
    obstacles: set[Coordinate]
    drone_position: Coordinate
    planned_path: list[Coordinate] | None
    replanned: bool
    goal_reached: bool
    emergency_detected: bool
    safe_stop: bool
    safe_stop_reason: str | None
    annotated_frame: np.ndarray


class VideoNavigationAgent:
    """Coordinate video perception, dynamic replanning and simulation."""

    def __init__(
        self,
        model_path: str = (
            "artifacts/training/visdrone_full_5epochs/weights/best.pt"
        ),
        confidence_threshold: float = 0.40,
        grid_rows: int = 30,
        grid_cols: int = 30,
    ) -> None:
        """Initialize the video navigation agent."""

        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError(
                "Confidence threshold must be between 0 and 1."
            )

        if grid_rows <= 0 or grid_cols <= 0:
            raise ValueError("Grid dimensions must be positive.")

        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.confidence_threshold = confidence_threshold

        self.detector = ObstacleDetector(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
        )

        self.simulator: DroneSimulator | None = None
        self.goal: Coordinate | None = None
        self.frame_results: list[VideoFrameResult] = []
        self.previous_obstacles: set[Coordinate] = set()

    def initialize_mission(
        self,
        start: Coordinate = (0, 0),
        goal: Coordinate | None = None,
    ) -> None:
        """Initialize the drone and generate its initial route."""

        if goal is None:
            goal = (self.grid_rows - 1, self.grid_cols - 1)

        static_grid = [
            [0 for _ in range(self.grid_cols)]
            for _ in range(self.grid_rows)
        ]

        initial_path = astar(static_grid, start, goal)

        if initial_path is None:
            raise RuntimeError(
                "Initial navigation path could not be generated."
            )

        self.simulator = DroneSimulator(
            grid=static_grid,
            path=initial_path,
        )

        self.goal = goal
        self.frame_results = []
        self.previous_obstacles = set()

    def process_frame(
        self,
        frame: np.ndarray,
        frame_index: int,
    ) -> VideoFrameResult:
        """Process one frame and update the simulated drone."""

        if (
            not isinstance(frame, np.ndarray)
            or frame.size == 0
            or frame.ndim != 3
            or frame.shape[2] != 3
        ):
            raise ValueError(
                "Input must be a valid three-channel video frame."
            )

        if self.simulator is None:
            raise RuntimeError(
                "Initialize the mission before processing frames."
            )

        if frame_index <= 0:
            raise ValueError("Frame index must be positive.")

        image_height, image_width = frame.shape[:2]

        # Step 1: Detect objects in the current frame.
        detections = self.detector.detect(frame)

        # Step 2: Convert detections into grid obstacles.
        obstacles = detections_to_grid_obstacles(
            detections=detections,
            image_width=image_width,
            image_height=image_height,
            grid_rows=self.grid_rows,
            grid_cols=self.grid_cols,
            confidence_threshold=self.confidence_threshold,
        )

        # Step 3: Check whether the obstacle configuration changed.
        changed = obstacles != self.previous_obstacles
        replanned = False

        if changed and not self.simulator.reached_goal:
            # Step 4: Replan from the drone's current position.
            self.simulator.replan(obstacles)
            replanned = True

        self.previous_obstacles = obstacles.copy()

        # Step 5: Move the drone by one simulation step.
        if (
            not self.simulator.reached_goal
            and not self.simulator.safe_stop
        ):
            self.simulator.step()

        # Step 6: Annotate the current frame.
        annotated_frame = self._annotate_frame(
            frame=frame,
            detections=detections,
            obstacles=obstacles,
        )

        # Step 7: Store the result.
        result = VideoFrameResult(
            frame_index=frame_index,
            detections=detections,
            obstacles=obstacles,
            drone_position=self.simulator.position,
            planned_path=(
                list(self.simulator.path)
                if self.simulator.path is not None
                else None
            ),
            replanned=replanned,
            goal_reached=self.simulator.reached_goal,
            emergency_detected=self.simulator.emergency_detected,
            safe_stop=self.simulator.safe_stop,
            safe_stop_reason=self.simulator.safe_stop_reason,
            annotated_frame=annotated_frame,
        )

        self.frame_results.append(result)

        return result

    def _annotate_frame(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        obstacles: set[Coordinate],
    ) -> np.ndarray:
        """Draw detections and navigation information on a frame."""

        annotated = frame.copy()

        # Draw detected object bounding boxes.
        for detection in detections:
            x1, y1, x2, y2 = (
                round(value)
                for value in detection.bounding_box
            )

            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            label = (
                f"{detection.label}: "
                f"{detection.confidence:.2f}"
            )

            cv2.putText(
                annotated,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        # Display drone navigation status.
        if self.simulator is not None:
            row, col = self.simulator.position

            status = (
                f"Drone: ({row}, {col}) | "
                f"Detections: {len(detections)} | "
                f"Obstacles: {len(obstacles)}"
            )

            cv2.putText(
                annotated,
                status,
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2,
            )

            if self.simulator.safe_stop:
                safety_status = "SAFE STOP"
            elif self.simulator.reached_goal:
                safety_status = "GOAL REACHED"
            else:
                safety_status = "NAVIGATING"

            cv2.putText(
                annotated,
                safety_status,
                (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255)
                if self.simulator.safe_stop
                else (255, 255, 0),
                2,
            )

        return annotated

    def process_video(
        self,
        video_path: str | Path,
        start: Coordinate = (0, 0),
        goal: Coordinate | None = None,
        frame_skip: int = 1,
        max_frames: int | None = None,
    ) -> Iterator[VideoFrameResult]:
        """Read and process video frames sequentially."""

        video_file = Path(video_path)

        if not video_file.is_file():
            raise FileNotFoundError(
                f"Video file not found: {video_file}"
            )

        if frame_skip < 1:
            raise ValueError("Frame skip must be at least 1.")

        if max_frames is not None and max_frames <= 0:
            raise ValueError("Maximum frames must be positive.")

        capture = cv2.VideoCapture(str(video_file))

        if not capture.isOpened():
            capture.release()
            raise ValueError(
                f"Could not open video: {video_file}"
            )

        self.initialize_mission(start=start, goal=goal)

        frame_index = 0
        processed_count = 0

        try:
            while True:
                success, frame = capture.read()

                if not success:
                    break

                frame_index += 1

                if (frame_index - 1) % frame_skip != 0:
                    continue

                result = self.process_frame(
                    frame=frame,
                    frame_index=frame_index,
                )

                processed_count += 1
                yield result

                if max_frames is not None and processed_count >= max_frames:
                    break

        finally:
            capture.release()


    def process_sequence(
        self,
        sequence_dir: str | Path,
        start: Coordinate = (0, 0),
        goal: Coordinate | None = None,
        frame_skip: int = 1,
        max_frames: int | None = None,
    ) -> Iterator[VideoFrameResult]:
        """Process a directory of sequential image frames."""

        sequence_path = Path(sequence_dir)

        if not sequence_path.is_dir():
            raise FileNotFoundError(
                f"Sequence directory not found: {sequence_path}"
            )

        if frame_skip < 1:
            raise ValueError("frame_skip must be at least 1")

        if max_frames is not None and max_frames < 1:
            raise ValueError("max_frames must be positive")

        frame_paths = sorted(
            sequence_path.glob("*.jpg"),
            key=lambda path: int(path.stem)
            if path.stem.isdigit()
            else path.stem,
        )

        if not frame_paths:
            raise ValueError(
                f"No JPG frames found in: {sequence_path}"
            )

        self.initialize_mission(start=start, goal=goal)

        processed_count = 0

        for frame_number, frame_path in enumerate(frame_paths, start=1):
            if (frame_number - 1) % frame_skip != 0:
                continue

            frame = cv2.imread(str(frame_path))

            if frame is None:
                continue

            result = self.process_frame(
                frame=frame,
                frame_index=frame_number,
            )

            processed_count += 1
            yield result

            if max_frames is not None and processed_count >= max_frames:
                break
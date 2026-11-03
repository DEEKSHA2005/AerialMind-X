"""Autonomous vision-based navigation agent for AerialMind-X."""

from __future__ import annotations

from dataclasses import dataclass

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
class AgentResult:
    """Store the results of an autonomous navigation mission."""

    detections: list[Detection]
    obstacles: set[Coordinate]
    initial_path: list[Coordinate]
    planned_path: list[Coordinate] | None
    visited_positions: list[Coordinate]
    final_position: Coordinate
    goal: Coordinate
    goal_reached: bool
    emergency_detected: bool
    safe_stop: bool
    safe_stop_reason: str | None
    decision: str
    annotated_image: np.ndarray


class AutonomousAgent:
    """Coordinate perception, planning, replanning and drone simulation."""

    def __init__(
        self,
        model_path: str = (
            "artifacts/training/visdrone_full_5epochs/weights/best.pt"
        ),
        confidence_threshold: float = 0.40,
        grid_rows: int = 30,
        grid_cols: int = 30,
    ) -> None:
        """Initialize the autonomous agent."""

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

    def run_mission(
        self,
        image: np.ndarray,
        start: Coordinate = (0, 0),
        goal: Coordinate | None = None,
    ) -> AgentResult:
        """Execute one complete image-based navigation mission."""

        if (
            not isinstance(image, np.ndarray)
            or image.size == 0
            or image.ndim != 3
            or image.shape[2] != 3
        ):
            raise ValueError(
                "Input must be a valid three-channel image."
            )

        if goal is None:
            goal = (self.grid_rows - 1, self.grid_cols - 1)

        image_height, image_width = image.shape[:2]

        # Step 1: Perceive obstacles.
        detections = self.detector.detect(image)

        # Step 2: Convert detections into grid obstacles.
        obstacles = detections_to_grid_obstacles(
            detections=detections,
            image_width=image_width,
            image_height=image_height,
            grid_rows=self.grid_rows,
            grid_cols=self.grid_cols,
            confidence_threshold=self.confidence_threshold,
        )

        # Step 3: Create the initial navigation grid.
        static_grid = [
            [0 for _ in range(self.grid_cols)]
            for _ in range(self.grid_rows)
        ]

        initial_path = astar(static_grid, start, goal)

        if initial_path is None:
            raise RuntimeError(
                "Initial path could not be generated."
            )

        # Step 4: Initialize the drone simulator.
        simulator = DroneSimulator(
            grid=static_grid,
            path=initial_path,
        )

        # Step 5: Replan using detected obstacles.
        planned_path = simulator.replan(obstacles)

        decision = "Route planned successfully."

        if planned_path is None:
            decision = (
                "Emergency stop: No safe route available."
                if simulator.emergency_detected
                else "Safe stop: No valid route available."
            )
        else:
            # Step 6: Execute simulated drone movement.
            for _ in simulator.run():
                pass

            if simulator.reached_goal:
                decision = "Mission completed. Goal reached."
            elif simulator.safe_stop:
                decision = (
                    f"Safe stop: {simulator.safe_stop_reason}"
                )

        # Step 7: Annotate detected objects.
        annotated_image = image.copy()

        for detection in detections:
            x1, y1, x2, y2 = detection.bounding_box

            cv2.rectangle(
                annotated_image,
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
                annotated_image,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        return AgentResult(
            detections=detections,
            obstacles=obstacles,
            initial_path=initial_path,
            planned_path=planned_path,
            visited_positions=simulator.visited_positions,
            final_position=simulator.position,
            goal=goal,
            goal_reached=simulator.reached_goal,
            emergency_detected=simulator.emergency_detected,
            safe_stop=simulator.safe_stop,
            safe_stop_reason=simulator.safe_stop_reason,
            decision=decision,
            annotated_image=annotated_image,
        )

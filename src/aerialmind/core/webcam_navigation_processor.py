"""Real-time webcam navigation processor for AerialMind-X."""

from __future__ import annotations

from threading import Lock
from time import perf_counter

import av
from streamlit_webrtc import VideoProcessorBase

from aerialmind.core.video_navigation_agent import VideoNavigationAgent


class WebcamNavigationProcessor(VideoProcessorBase):
    """Process live webcam frames using the existing navigation agent."""

    def __init__(
        self,
        confidence_threshold: float = 0.40,
        grid_size: int = 30,
    ) -> None:
        """Initialize an independent webcam navigation session."""

        self.agent = VideoNavigationAgent(
            confidence_threshold=confidence_threshold,
            grid_rows=grid_size,
            grid_cols=grid_size,
        )

        self.agent.initialize_mission()

        self.frame_index = 0
        self.total_detections = 0
        self.route_updates = 0
        self.path_changes = 0
        self.total_processing_time = 0.0

        self.previous_path = (
            list(self.agent.simulator.path)
            if self.agent.simulator is not None
            else None
        )
        
        self.latest_result = None
        self.error = None

        self.visited_positions = []
        self.obstacle_cells = set()

        self.lock = Lock()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Process one incoming webcam frame."""

        image = frame.to_ndarray(format="bgr24")

        try:
            with self.lock:
                self.frame_index += 1

                start_time = perf_counter()

                result = self.agent.process_frame(
                    frame=image,
                    frame_index=self.frame_index,
                )

                processing_time = perf_counter() - start_time

                self.total_processing_time += processing_time
                self.total_detections += len(result.detections)

                current_path = (
                    list(result.planned_path)
                    if result.planned_path is not None
                    else None
                )

                if result.replanned:
                    self.route_updates += 1

                    if (
                        current_path is not None
                        and current_path != self.previous_path
                    ):
                        self.path_changes += 1

                if current_path is not None:
                    self.previous_path = current_path
                self.visited_positions.append(
                    result.drone_position
                )

                self.obstacle_cells = set(result.obstacles)

                self.latest_result = result
                self.error = None

                # Prevent unlimited frame history growth.
                self.agent.frame_results.clear()

                return av.VideoFrame.from_ndarray(
                    result.annotated_frame,
                    format="bgr24",
                )

        except (ValueError, TypeError, RuntimeError) as error:
            with self.lock:
                self.error = str(error)
            return frame

    def get_status(self) -> dict:
        """Return a thread-safe snapshot of webcam navigation metrics."""

        with self.lock:
            result = self.latest_result

            return {
                "frame_index": self.frame_index,
                "total_detections": self.total_detections,
                "route_updates": self.route_updates,
                "path_changes": self.path_changes,
                "average_processing_time": (
                    self.total_processing_time / self.frame_index
                    if self.frame_index
                    else 0.0
                ),
                "drone_position": (
                    result.drone_position if result else None
                ),
                "planned_path": (
                    list(result.planned_path)
                    if result and result.planned_path is not None
                    else None
                ),
                "detections": (
                    list(result.detections) if result else []
                ),
                "obstacles": (
                    set(result.obstacles) if result else set()
                ),
                "visited_positions": list(self.visited_positions),
                "obstacle_cells": set(self.obstacle_cells),
                "goal_reached": (
                    result.goal_reached if result else False
                ),
                "emergency_detected": (
                    result.emergency_detected if result else False
                ),
                "safe_stop": result.safe_stop if result else False,
                "safe_stop_reason": (
                    result.safe_stop_reason if result else None
                ),
                "error": self.error,
            }

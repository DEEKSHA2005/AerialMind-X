"""Evaluation metrics for AerialMind-X navigation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import pairwise

from aerialmind.planning.astar import Coordinate


def calculate_path_cost(path: Sequence[Coordinate]) -> float:
    """Calculate path cost using 1.0 for straight and sqrt(2) for diagonal moves."""
    if not path:
        raise ValueError("Path must not be empty.")

    total_cost = 0.0

    for current, following in pairwise(path):
        row_delta = abs(following[0] - current[0])
        col_delta = abs(following[1] - current[1])

        if row_delta > 1 or col_delta > 1 or (row_delta == 0 and col_delta == 0):
            raise ValueError("Consecutive path coordinates must be adjacent.")

        if row_delta == 1 and col_delta == 1:
            total_cost += math.sqrt(2)
        else:
            total_cost += 1.0

    return total_cost


def calculate_success_rate(successes: Sequence[bool]) -> float:
    """Return the percentage of successful scenarios, from 0 to 100."""
    if not successes:
        raise ValueError("At least one scenario result is required.")

    return sum(successes) / len(successes) * 100.0


def calculate_goal_reaching_rate(
    final_positions: Sequence[Coordinate],
    goal: Coordinate,
) -> float:
    """Return the percentage of simulations that reached the goal."""
    if not final_positions:
        raise ValueError("At least one final position is required.")

    return sum(position == goal for position in final_positions) / len(
        final_positions
    ) * 100.0


def calculate_average_planning_time(times_seconds: Sequence[float]) -> float:
    """Calculate average planning time in seconds."""
    if not times_seconds:
        raise ValueError("At least one planning time is required.")

    if any(not math.isfinite(value) or value < 0 for value in times_seconds):
        raise ValueError("Planning times must be finite and non-negative.")

    return sum(times_seconds) / len(times_seconds)


def calculate_average_detection_time(times_seconds: Sequence[float]) -> float:
    """Calculate average object detection time in seconds."""
    return calculate_average_planning_time(times_seconds)


def calculate_average_replanning_time(times_seconds: Sequence[float]) -> float:
    """Calculate average dynamic replanning time in seconds."""
    return calculate_average_planning_time(times_seconds)


def calculate_processing_fps(
    frames_processed: int,
    elapsed_seconds: float,
) -> float:
    """Calculate video processing frames per second."""
    if frames_processed < 0:
        raise ValueError("Frames processed must be non-negative.")

    if not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0:
        raise ValueError("Elapsed time must be finite and positive.")

    return frames_processed / elapsed_seconds


def calculate_navigation_efficiency(
    optimal_path_cost: float,
    actual_path_cost: float,
) -> float:
    """Calculate navigation efficiency as a percentage."""
    costs = (optimal_path_cost, actual_path_cost)

    if any(not math.isfinite(cost) or cost < 0 for cost in costs):
        raise ValueError("Path costs must be finite and non-negative.")

    if actual_path_cost <= 0:
        raise ValueError("Actual path cost must be positive.")

    if optimal_path_cost > actual_path_cost:
        raise ValueError("Optimal path cost cannot exceed actual path cost.")

    return (optimal_path_cost / actual_path_cost) * 100.0


def calculate_average_mission_completion_time(
    times_seconds: Sequence[float],
) -> float:
    """Calculate average mission completion time in seconds."""
    return calculate_average_planning_time(times_seconds)

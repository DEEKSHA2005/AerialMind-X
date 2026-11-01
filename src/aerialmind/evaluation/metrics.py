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

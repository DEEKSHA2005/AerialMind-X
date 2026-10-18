
"""A* path planning on a two-dimensional occupancy grid."""

from __future__ import annotations

import heapq
import math
from itertools import count

Coordinate = tuple[int, int]


def _heuristic(a: Coordinate, b: Coordinate) -> float:
    """Estimate the remaining path cost using octile distance."""
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])

    diagonal_steps = min(dx, dy)
    straight_steps = max(dx, dy) - diagonal_steps

    return diagonal_steps * math.sqrt(2) + straight_steps


def astar(
    grid: list[list[int]],
    start: Coordinate,
    goal: Coordinate,
) -> list[Coordinate] | None:
    """Find a least-cost path through a rectangular occupancy grid.

    Grid convention:
        0 = free cell
        1 = blocked cell

    Coordinates use (row, column), with (0, 0) at the top-left.

    Movement:
        Horizontal and vertical moves cost 1.0.
        Diagonal moves cost sqrt(2).
        Diagonal corner-cutting is not allowed.

    Returns:
        A list of coordinates including start and goal, or None if no
        route exists.

    Raises:
        ValueError: If the grid, coordinates, or endpoints are invalid.
    """
    if not grid or not grid[0]:
        raise ValueError("Grid must not be empty.")

    width = len(grid[0])

    if any(len(row) != width for row in grid):
        raise ValueError("Grid must be rectangular.")

    if any(cell not in (0, 1) for row in grid for cell in row):
        raise ValueError("Grid cells must be 0 (free) or 1 (blocked).")

    rows, cols = len(grid), width

    def validate_coordinate(point: Coordinate, name: str) -> None:
        if (
            not isinstance(point, tuple)
            or len(point) != 2
            or any(type(value) is not int for value in point)
        ):
            raise ValueError(f"{name} must be a tuple of two integers.")

        row, col = point

        if not (0 <= row < rows and 0 <= col < cols):
            raise ValueError(f"{name} is outside the grid.")

    validate_coordinate(start, "start")
    validate_coordinate(goal, "goal")

    if grid[start[0]][start[1]] == 1:
        raise ValueError("Start cell is blocked.")

    if grid[goal[0]][goal[1]] == 1:
        raise ValueError("Goal cell is blocked.")

    if start == goal:
        return [start]

    # Eight-direction movement is compulsory.
    # Straight moves cost 1.0; diagonal moves cost sqrt(2).
    directions = [
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, math.sqrt(2)),
        (-1, 1, math.sqrt(2)),
        (1, -1, math.sqrt(2)),
        (1, 1, math.sqrt(2)),
    ]

    frontier: list[tuple[float, int, Coordinate]] = []
    sequence = count()

    heapq.heappush(
        frontier,
        (_heuristic(start, goal), next(sequence), start),
    )

    came_from: dict[Coordinate, Coordinate] = {}
    cost_so_far: dict[Coordinate, float] = {start: 0.0}

    while frontier:
        _, _, current = heapq.heappop(frontier)

        if current == goal:
            path = [goal]

            while path[-1] != start:
                path.append(came_from[path[-1]])

            path.reverse()
            return path

        for dr, dc, move_cost in directions:
            nr = current[0] + dr
            nc = current[1] + dc

            # Skip positions outside the grid.
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue

            # Skip blocked destination cells.
            if grid[nr][nc] == 1:
                continue

            # Prevent diagonal movement through blocked corners.
            if dr != 0 and dc != 0:
                if grid[current[0] + dr][current[1]] == 1:
                    continue

                if grid[current[0]][current[1] + dc] == 1:
                    continue

            neighbor = (nr, nc)
            new_cost = cost_so_far[current] + move_cost

            if new_cost < cost_so_far.get(neighbor, math.inf):
                cost_so_far[neighbor] = new_cost
                came_from[neighbor] = current

                priority = new_cost + _heuristic(neighbor, goal)

                heapq.heappush(
                    frontier,
                    (priority, next(sequence), neighbor),
                )

    return None

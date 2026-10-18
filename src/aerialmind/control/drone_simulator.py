
"""Grid-based drone movement simulation for AerialMind-X."""

from __future__ import annotations

from collections.abc import Iterator
from itertools import pairwise

Coordinate = tuple[int, int]


class DroneSimulator:
    """Simulate a drone following a planned grid path."""

    def __init__(
        self,
        grid: list[list[int]],
        path: list[Coordinate],
    ) -> None:
        """Initialize the simulator with a grid and planned path."""
        if not grid or not grid[0]:
            raise ValueError("Grid must not be empty.")

        width = len(grid[0])

        if any(len(row) != width for row in grid):
            raise ValueError("Grid must be rectangular.")

        if any(cell not in (0, 1) for row in grid for cell in row):
            raise ValueError("Grid cells must be 0 (free) or 1 (blocked).")

        if not path:
            raise ValueError("Path must not be empty.")

        self.grid = grid
        self.path = path
        self.rows = len(grid)
        self.cols = width

        for point in path:
            if (
                not isinstance(point, tuple)
                or len(point) != 2
                or any(type(value) is not int for value in point)
            ):
                raise ValueError("Path coordinates must be integer tuples.")

            row, col = point

            if not (0 <= row < self.rows and 0 <= col < self.cols):
                raise ValueError("Path coordinate is outside the grid.")

            if grid[row][col] == 1:
                raise ValueError("Path cannot pass through blocked cells.")

        if len(path) > 1 and any(
            abs(path[index + 1][0] - path[index][0]) > 1
            or abs(path[index + 1][1] - path[index][1]) > 1
            or path[index] == path[index + 1]
            for index in range(len(path) - 1)
        ):
            raise ValueError("Consecutive path positions must be adjacent.")

        for current, following in pairwise(path):
            row, col = current
            next_row, next_col = following

            if (
                next_row != row
                and next_col != col
                and (
                    grid[next_row][col] == 1
                    or grid[row][next_col] == 1
                )
            ):
                raise ValueError(
                    "Diagonal movement cannot cut blocked corners."
                )
        

        self.position = path[0]
        self.visited_positions: list[Coordinate] = [self.position]

    @property
    def goal(self) -> Coordinate:
        """Return the final destination."""
        return self.path[-1]

    @property
    def reached_goal(self) -> bool:
        """Return whether the drone has reached its destination."""
        return self.position == self.goal

    def step(self) -> Coordinate:
        """Move one position along the planned path."""
        if self.reached_goal:
            return self.position

        current_index = len(self.visited_positions) - 1
        next_position = self.path[current_index + 1]

        self.position = next_position
        self.visited_positions.append(next_position)

        return self.position

    def run(self) -> Iterator[Coordinate]:
        """Yield the drone's position after each movement."""
        while not self.reached_goal:
            yield self.step()

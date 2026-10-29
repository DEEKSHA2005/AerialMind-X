"""Grid-based drone movement with dynamic path replanning."""

from __future__ import annotations

from collections.abc import Iterable
from itertools import pairwise

from aerialmind.planning.astar import Coordinate
from aerialmind.planning.dynamic_replanner import DynamicReplanner


class DroneSimulator:
    """Simulate drone movement and safely replan around obstacles."""

    def __init__(
        self,
        grid: list[list[int]],
        path: list[Coordinate],
    ) -> None:
        """Initialize the simulator with a grid and valid planned path."""
        if not grid or not grid[0]:
            raise ValueError("Grid must not be empty.")

        width = len(grid[0])

        if any(len(row) != width for row in grid):
            raise ValueError("Grid must be rectangular.")

        if any(cell not in (0, 1) for row in grid for cell in row):
            raise ValueError("Grid cells must be 0 (free) or 1 (blocked).")

        if not path:
            raise ValueError("Path must not be empty.")

        self.static_grid = [row[:] for row in grid]
        self.grid = [row[:] for row in grid]
        self.rows = len(grid)
        self.cols = width
        self._goal = path[-1]
        self.path: list[Coordinate] | None = list(path)
        self._path_index = 0
        self.safe_stop = False

        for point in path:
            self._validate_coordinate(point, "Path coordinate")

            row, col = point
            if self.grid[row][col] == 1:
                raise ValueError("Path cannot pass through blocked cells.")

        if len(path) > 1 and any(
            abs(following[0] - current[0]) > 1
            or abs(following[1] - current[1]) > 1
            or current == following
            for current, following in pairwise(path)
        ):
            raise ValueError("Consecutive path positions must be adjacent.")

        for current, following in pairwise(path):
            row, col = current
            next_row, next_col = following

            if (
                next_row != row
                and next_col != col
                and (
                    self.grid[next_row][col] == 1
                    or self.grid[row][next_col] == 1
                )
            ):
                raise ValueError(
                    "Diagonal movement cannot cut blocked corners."
                )

        self.position = path[0]
        self.visited_positions: list[Coordinate] = [self.position]

    def _validate_coordinate(self, point: Coordinate, name: str) -> None:
        """Check that a coordinate is valid and inside the grid."""
        if (
            not isinstance(point, tuple)
            or len(point) != 2
            or any(type(value) is not int for value in point)
        ):
            raise ValueError(f"{name} must be a tuple of two integers.")

        row, col = point

        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise ValueError(f"{name} is outside the grid.")

    @property
    def goal(self) -> Coordinate:
        """Return the original destination."""
        return self._goal

    @property
    def reached_goal(self) -> bool:
        """Return whether the drone has reached its destination."""
        return self.position == self.goal

    def replan(self, obstacles: Iterable[Coordinate]) -> list[Coordinate] | None:
        """Update dynamic obstacles and find a safe path from current position.

        If the route is blocked or no safe route exists, the simulator
        enters safe-stop state and will not continue along the old path.
        """
        planner = DynamicReplanner(
            grid=self.static_grid,
            start=self.position,
            goal=self.goal,
        )

        new_path = planner.update(
            current_position=self.position,
            obstacles=obstacles,
        )

        self.grid = planner.grid
        self.path = new_path
        self._path_index = 0
        self.safe_stop = planner.safe_stop

        return new_path

    def step(self) -> Coordinate:
        """Move one step, refusing unsafe or unavailable movements."""
        if self.reached_goal or self.safe_stop or self.path is None:
            return self.position

        if self._path_index + 1 >= len(self.path):
            return self.position

        next_position = self.path[self._path_index + 1]
        row, col = self.position
        next_row, next_col = next_position

        # Verify the next cell is still free.
        if self.grid[next_row][next_col] == 1:
            self.safe_stop = True
            return self.position

        # Prevent diagonal corner-cutting after a grid update.
        if (
            next_row != row
            and next_col != col
            and (
                self.grid[next_row][col] == 1
                or self.grid[row][next_col] == 1
            )
        ):
            self.safe_stop = True
            return self.position

        self.position = next_position
        self._path_index += 1
        self.visited_positions.append(next_position)

        return self.position

    def run(self):
        """Yield each new position until reaching the goal or stopping."""
        while not self.reached_goal and not self.safe_stop:
            previous_position = self.position
            next_position = self.step()

            if next_position == previous_position:
                break

            yield next_position

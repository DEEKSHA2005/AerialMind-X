"""Dynamic obstacle updates and goal-directed path replanning."""

from __future__ import annotations

from collections.abc import Iterable

from aerialmind.planning.astar import Coordinate, astar


class DynamicReplanner:
    """Recalculate a safe route when dynamic obstacles change."""

    def __init__(
        self,
        grid: list[list[int]],
        start: Coordinate,
        goal: Coordinate,
    ) -> None:
        """Initialize the planner with a static occupancy grid."""
        if not grid or not grid[0]:
            raise ValueError("Grid must not be empty.")

        width = len(grid[0])
        if any(len(row) != width for row in grid):
            raise ValueError("Grid must be rectangular.")

        if any(cell not in (0, 1) for row in grid for cell in row):
            raise ValueError("Grid cells must be 0 (free) or 1 (blocked).")

        self.static_grid = [row[:] for row in grid]
        self.rows = len(grid)
        self.cols = width
        self.goal = goal
        self.current_position = start
        self.dynamic_obstacles: set[Coordinate] = set()
        self.grid = [row[:] for row in grid]
        self.path: list[Coordinate] | None = None
        self.safe_stop = False

        # Validate endpoints and find the initial route.
        self.path = astar(self.grid, start, goal)
        self.safe_stop = self.path is None

    def _validate_coordinate(self, point: Coordinate, name: str) -> None:
        """Validate that a coordinate lies inside the grid."""
        if (
            not isinstance(point, tuple)
            or len(point) != 2
            or any(type(value) is not int for value in point)
        ):
            raise ValueError(f"{name} must be a tuple of two integers.")

        row, col = point
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise ValueError(f"{name} is outside the grid.")

    def update(
        self,
        current_position: Coordinate,
        obstacles: Iterable[Coordinate],
    ) -> list[Coordinate] | None:
        """Update dynamic obstacles and calculate a new route.

        Returns a path from the current position to the goal, or None
        when no safe route can be found.
        """
        self._validate_coordinate(current_position, "Current position")
        updated_obstacles = set(obstacles)

        for obstacle in updated_obstacles:
            self._validate_coordinate(obstacle, "Obstacle")

        self.current_position = current_position
        self.dynamic_obstacles = updated_obstacles

        # Rebuild from the static grid so removed obstacles become free.
        self.grid = [row[:] for row in self.static_grid]

        for row, col in self.dynamic_obstacles:
            self.grid[row][col] = 1

        # Never plan through the drone's current position or destination
        # if either is reported as blocked.
        if (
            self.grid[current_position[0]][current_position[1]] == 1
            or self.grid[self.goal[0]][self.goal[1]] == 1
        ):
            self.path = None
            self.safe_stop = True
            return None

        self.path = astar(self.grid, current_position, self.goal)
        self.safe_stop = self.path is None

        return self.path

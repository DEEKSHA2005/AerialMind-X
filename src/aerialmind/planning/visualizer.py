
"""Visualization utilities for AerialMind-X path planning."""

from __future__ import annotations

import matplotlib.pyplot as plt


def visualize_path(
    grid: list[list[int]],
    path: list[tuple[int, int]] | None,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> None:
    """Display the occupancy grid and the planned path.

    Grid convention:
        0 = free cell
        1 = blocked cell

    Coordinates use (row, column), with (0, 0) at the top-left.
    """
    if not grid or not grid[0]:
        raise ValueError("Grid must not be empty.")

    width = len(grid[0])

    if any(len(row) != width for row in grid):
        raise ValueError("Grid must be rectangular.")

    if any(cell not in (0, 1) for row in grid for cell in row):
        raise ValueError("Grid cells must be 0 (free) or 1 (blocked).")

    rows, cols = len(grid), width

    def validate_coordinate(point: tuple[int, int], name: str) -> None:
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

    if path is not None:
        for point in path:
            validate_coordinate(point, "path coordinate")

            if grid[point[0]][point[1]] == 1:
                raise ValueError("Path cannot pass through blocked cells.")

        if not path or path[0] != start or path[-1] != goal:
            raise ValueError("Path must begin at start and end at goal.")

    figure, axis = plt.subplots(figsize=(8, 8))

    axis.imshow(
        grid,
        cmap="Greys",
        origin="upper",
        vmin=0,
        vmax=1,
    )

    axis.set_xticks(range(cols))
    axis.set_yticks(range(rows))
    axis.set_xticks([x - 0.5 for x in range(cols + 1)], minor=True)
    axis.set_yticks([y - 0.5 for y in range(rows + 1)], minor=True)
    axis.grid(which="minor", color="lightgray", linewidth=1)
    axis.tick_params(which="minor", bottom=False, left=False)

    if path:
        path_rows = [point[0] for point in path]
        path_cols = [point[1] for point in path]

        axis.plot(
            path_cols,
            path_rows,
            color="dodgerblue",
            linewidth=2.5,
            marker="o",
            markersize=5,
            label="Planned path",
        )

    axis.scatter(
        start[1],
        start[0],
        color="limegreen",
        edgecolors="black",
        s=160,
        zorder=3,
        label="Start",
    )

    axis.scatter(
        goal[1],
        goal[0],
        color="red",
        edgecolors="black",
        s=160,
        zorder=3,
        label="Goal",
    )

    axis.set_title("AerialMind-X: A* Path Planning")
    axis.set_xlabel("Column")
    axis.set_ylabel("Row")
    axis.legend(loc="upper right")
    axis.set_aspect("equal")

    figure.tight_layout()
    plt.show()

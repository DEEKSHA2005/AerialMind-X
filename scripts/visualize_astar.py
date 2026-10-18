
"""Run a demonstration of AerialMind-X A* path visualization."""

from aerialmind.planning.astar import astar
from aerialmind.planning.visualizer import visualize_path


def main() -> None:
    grid = [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 1, 1, 0],
        [0, 1, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 0, 0, 0],
    ]

    start = (0, 0)
    goal = (7, 7)

    path = astar(grid, start, goal)

    if path is None:
        print("No valid route found.")
    else:
        print(f"Start: {start}")
        print(f"Goal: {goal}")
        print(f"Path coordinates: {path}")
        print(f"Number of cells in path: {len(path)}")

    visualize_path(grid, path, start, goal)


if __name__ == "__main__":
    main()

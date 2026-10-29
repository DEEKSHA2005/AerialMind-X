"""Run AerialMind-X with vision detection and dynamic replanning."""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from aerialmind.control.drone_simulator import DroneSimulator
from aerialmind.perception.obstacle_detector import ObstacleDetector
from aerialmind.perception.vision_grid_mapper import (
    detections_to_grid_obstacles,
)
from aerialmind.planning.astar import astar


def main() -> None:
    """Run perception, planning, dynamic replanning, and simulation."""
    image_path = Path("data/samples/obstacle_scene.jpg")
    grid_rows = 8
    grid_cols = 8
    start = (0, 0)
    goal = (7, 7)

    if not image_path.is_file():
        raise FileNotFoundError(f"Sample image not found: {image_path}")

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    image_height, image_width = image.shape[:2]

    print("=" * 60)
    print("AerialMind-X: Adaptive Drone Navigation")
    print("=" * 60)

    # 1. Detect obstacles in the camera image.
    detector = ObstacleDetector(
        model_path="yolo11n.pt",
        confidence_threshold=0.25,
    )
    detections = detector.detect(image_path)

    print(f"\nImage: {image_path}")
    print(f"Objects detected: {len(detections)}")

    for detection in detections:
        print(
            f"  {detection.label}: "
            f"confidence={detection.confidence:.2f}, "
            f"box={detection.bounding_box}"
        )

    # 2. Convert image detections into simulated grid coordinates.
    vision_obstacles = detections_to_grid_obstacles(
        detections=detections,
        image_width=image_width,
        image_height=image_height,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        confidence_threshold=0.25,
    )

    # This is a simplified simulation grid, not a calibrated world map.
    grid = [
        [0 for _ in range(grid_cols)]
        for _ in range(grid_rows)
    ]

    for row, col in vision_obstacles:
        grid[row][col] = 1

    # Keep start and goal available for this simulation.
    grid[start[0]][start[1]] = 0
    grid[goal[0]][goal[1]] = 0

    # 3. Calculate the initial route.
    initial_path = astar(grid, start, goal)

    if initial_path is None:
        print("\nNo initial route found.")
        return

    drone = DroneSimulator(grid, initial_path)

    print(f"\nVision-derived obstacle cells: {sorted(vision_obstacles)}")
    print(f"Initial path: {initial_path}")

    # 4. Move two steps before introducing the simulated obstacle.
    print("\nInitial drone movement:")

    for _ in range(2):
        previous_position = drone.position
        new_position = drone.step()

        if new_position == previous_position:
            break

        print(f"  Drone moved to: {new_position}")

    if drone.reached_goal:
        print("\nDrone reached the goal before the event.")
        return

    if drone.safe_stop:
        print("\nDrone entered safe-stop mode before the event.")
        return

    # 5. Introduce a simulated obstacle on the next route cell.
    assert drone.path is not None
    next_cell = drone.path[drone._path_index + 1]

    # Retain the vision-derived obstacles and add the new simulated one.
    dynamic_obstacles = set(vision_obstacles)
    dynamic_obstacles.add(next_cell)

    print("\nSIMULATED DYNAMIC OBSTACLE EVENT")
    print(f"New obstacle cell: {next_cell}")
    print(f"Drone's current position: {drone.position}")

    # 6. Replan from the drone's current position.
    previous_remaining_path = drone.path[drone._path_index:]
    updated_path = drone.replan(dynamic_obstacles)

    if updated_path is None:
        print("No safe alternative route exists.")
        print("Drone entered safe-stop mode.")
    else:
        print(f"Previous remaining route: {previous_remaining_path}")
        print(f"Updated route: {updated_path}")

        route_changed = updated_path != previous_remaining_path
        print(f"Route changed: {route_changed}")

        if next_cell in updated_path[1:]:
            print("WARNING: Updated route still includes the new obstacle.")
        else:
            print("Updated route avoids the new obstacle.")

        print("\nContinuing drone movement:")

        for position in drone.run():
            print(f"  Drone moved to: {position}")

    # 7. Visualize the final occupancy grid and navigation history.
    figure, axis = plt.subplots(figsize=(8, 8))

    axis.imshow(
        np.array(drone.grid),
        cmap="Greys",
        origin="upper",
        vmin=0,
        vmax=1,
    )

    initial_rows = [point[0] for point in initial_path]
    initial_cols = [point[1] for point in initial_path]

    axis.plot(
        initial_cols,
        initial_rows,
        "--",
        marker=".",
        label="Initial route",
    )

    if updated_path is not None:
        updated_rows = [point[0] for point in updated_path]
        updated_cols = [point[1] for point in updated_path]

        axis.plot(
            updated_cols,
            updated_rows,
            "-",
            marker=".",
            label="Replanned route",
        )

    visited_rows = [point[0] for point in drone.visited_positions]
    visited_cols = [point[1] for point in drone.visited_positions]

    axis.plot(
        visited_cols,
        visited_rows,
        "-",
        marker="o",
        label="Drone trajectory",
    )

    obstacle_rows = [point[0] for point in dynamic_obstacles]
    obstacle_cols = [point[1] for point in dynamic_obstacles]

    axis.scatter(
        obstacle_cols,
        obstacle_rows,
        marker="X",
        s=100,
        label="Obstacle cells",
    )

    axis.scatter(
        [start[1]], [start[0]],
        marker="s", s=100, label="Start"
    )
    axis.scatter(
        [goal[1]], [goal[0]],
        marker="*", s=180, label="Goal"
    )

    axis.set_title("AerialMind-X: Dynamic Obstacle Replanning")
    axis.set_xlabel("Grid column")
    axis.set_ylabel("Grid row")
    axis.set_xticks(range(grid_cols))
    axis.set_yticks(range(grid_rows))
    axis.grid(True, alpha=0.3)
    axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    figure.tight_layout()
    plt.show()

    print("\nFINAL SIMULATION SUMMARY")
    print(f"Final position: {drone.position}")
    print(f"Goal reached: {drone.reached_goal}")
    print(f"Safe stop: {drone.safe_stop}")
    print(f"Total movements: {len(drone.visited_positions) - 1}")


if __name__ == "__main__":
    main()

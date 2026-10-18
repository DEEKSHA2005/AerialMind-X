
"""Demonstrate drone movement along an A* planned path."""

from aerialmind.control.drone_simulator import DroneSimulator
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

    # Calculate the route using the A* planner.
    path = astar(grid, start, goal)

    if path is None:
        print("No valid route found.")
        return

    # Initialize the simulated drone.
    drone = DroneSimulator(grid, path)

    print(f"Start position: {drone.position}")
    print(f"Goal position: {drone.goal}")
    print(f"Planned path: {path}")
    print("\nDrone movement:")

    # Move the drone one step at a time.
    for position in drone.run():
        print(f"Drone moved to: {position}")

    print("\nSimulation completed.")
    print(f"Final position: {drone.position}")
    print(f"Goal reached: {drone.reached_goal}")
    print(f"Total movements: {len(drone.visited_positions) - 1}")

    # Display the planned route.
    visualize_path(grid, path, start, goal)


if __name__ == "__main__":
    main()

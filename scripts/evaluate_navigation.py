"""Evaluate AerialMind-X planning and dynamic navigation performance."""

from __future__ import annotations

import time

from aerialmind.control.drone_simulator import DroneSimulator
from aerialmind.evaluation.metrics import (
    calculate_average_planning_time,
    calculate_goal_reaching_rate,
    calculate_path_cost,
    calculate_success_rate,
)
from aerialmind.planning.astar import astar


def run_evaluation() -> None:
    """Run controlled navigation scenarios and print evaluation metrics."""
    grid_size = 8
    base_grid = [[0] * grid_size for _ in range(grid_size)]
    start = (0, 0)
    goal = (7, 7)

    # Each scenario introduces a different obstacle condition.
    scenarios = [
        {"name": "Route-blocking obstacle", "obstacles": {(3, 3)}},
        {"name": "Multiple obstacles", "obstacles": {(2, 2), (3, 3), (4, 4)}},
        {
            "name": "Complete barrier",
            "obstacles": {(row, 3) for row in range(grid_size)},
        },
    ]

    initial_times: list[float] = []
    replanning_times: list[float] = []
    goal_results: list[bool] = []
    replanning_results: list[bool] = []
    safe_stop_results: list[bool] = []

    print("=" * 64)
    print("AerialMind-X Navigation Evaluation")
    print("=" * 64)

    for scenario in scenarios:
        print(f"\nScenario: {scenario['name']}")

        # Measure initial A* planning time.
        started = time.perf_counter()
        initial_path = astar(base_grid, start, goal)
        initial_elapsed = time.perf_counter() - started
        initial_times.append(initial_elapsed)

        if initial_path is None:
            print("Initial planning failed.")
            goal_results.append(False)
            replanning_results.append(False)
            safe_stop_results.append(True)
            continue

        initial_cost = calculate_path_cost(initial_path)
        drone = DroneSimulator(base_grid, initial_path)

        # Move the drone two steps before the obstacle update.
        drone.step()
        drone.step()

        # Measure dynamic replanning time from the current position.
        started = time.perf_counter()
        updated_path = drone.replan(scenario["obstacles"])
        replanning_elapsed = time.perf_counter() - started
        replanning_times.append(replanning_elapsed)

        if updated_path is None:
            updated_cost = None
            replan_success = False
        else:
            updated_cost = calculate_path_cost(updated_path)
            replan_success = True
            list(drone.run())

        reached_goal = drone.reached_goal
        safe_stop = drone.safe_stop

        goal_results.append(reached_goal)
        replanning_results.append(replan_success)
        safe_stop_results.append(safe_stop)

        print(f"Initial path cost: {initial_cost:.3f}")
        print(f"Initial planning time: {initial_elapsed:.6f} seconds")
        print(f"Replanning time: {replanning_elapsed:.6f} seconds")
        print(
            "Updated path cost: "
            f"{updated_cost:.3f}" if updated_cost is not None
            else "Updated path cost: No route"
        )
        print(f"Replanning succeeded: {replan_success}")
        print(f"Goal reached: {reached_goal}")
        print(f"Safe stop: {safe_stop}")

    print("\n" + "=" * 64)
    print("AGGREGATE RESULTS")
    print("=" * 64)
    goal_reaching_rate = calculate_goal_reaching_rate(
        [goal if result else start for result in goal_results],
        goal,
    )

    print(f"Goal-reaching rate: {goal_reaching_rate:.1f}%")
    print(
        "Replanning success rate: "
        f"{calculate_success_rate(replanning_results):.1f}%"
    )
    print(
        "Average initial planning time: "
        f"{calculate_average_planning_time(initial_times):.6f} seconds"
    )
    print(
        "Average replanning time: "
        f"{calculate_average_planning_time(replanning_times):.6f} seconds"
    )
    print(
        "Safe-stop scenarios: "
        f"{sum(safe_stop_results)}/{len(safe_stop_results)}"
    )


if __name__ == "__main__":
    run_evaluation()

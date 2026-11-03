"""Visualize AerialMind-X navigation and evaluation results."""

from __future__ import annotations

import time

from aerialmind.control.drone_simulator import DroneSimulator
from aerialmind.evaluation.metrics import (
    calculate_goal_reaching_rate,
    calculate_path_cost,
    calculate_success_rate,
)
from aerialmind.evaluation.visualizer import (
    visualize_navigation_comparison,
    visualize_performance_dashboard,
)
from aerialmind.planning.astar import astar


def main() -> None:
    """Run normal navigation and emergency evaluation scenarios."""
    grid_size = 8
    grid = [[0] * grid_size for _ in range(grid_size)]

    start = (0, 0)
    goal = (7, 7)

    # Normal scenarios: all should have a feasible alternative route.
    navigation_scenarios = [
        {
            "name": "Single obstacle",
            "obstacles": {(3, 3)},
        },
        {
            "name": "Multiple obstacles",
            "obstacles": {(3, 3), (4, 4), (5, 5)},
        },
        {
            "name": "Obstacle cluster",
            "obstacles": {(3, 3), (3, 4), (4, 3), (4, 4)},
        },
    ]

    # Emergency scenarios are evaluated separately.
    emergency_scenarios = [
        {
            "name": "Obstacle at drone position",
            "obstacles": {(2, 2)},
        },
    ]

    unreachable_scenarios = [
        {
            "name": "Complete barrier",
            "obstacles": {(row, 3) for row in range(grid_size)},
        },
    ]

    scenario_names: list[str] = []  
    initial_costs: list[float] = []
    updated_costs: list[float] = []
    initial_times: list[float] = []
    replanning_times: list[float] = []

    final_positions: list[tuple[int, int]] = []
    replanning_results: list[bool] = []

    emergency_results: list[bool] = []
    unreachable_results: list[bool] = []
    emergency_safe_stops = 0
    unreachable_safe_stops = 0

    print("=" * 60)
    print("AerialMind-X UC7: Results Visualization")
    print("=" * 60)

    # Evaluate normal navigation scenarios.
    for scenario in navigation_scenarios:
        name = scenario["name"]
        obstacles = scenario["obstacles"]

        print(f"\nRunning navigation scenario: {name}")

        started = time.perf_counter()
        original_path = astar(grid, start, goal)
        initial_elapsed = time.perf_counter() - started

        if original_path is None:
            print("Initial route could not be generated.")
            continue

        initial_cost = calculate_path_cost(original_path)

        drone = DroneSimulator(grid, original_path)

        # Move the drone to (2, 2).
        drone.step()
        drone.step()

        # Replan around the newly detected obstacles.
        started = time.perf_counter()
        updated_path = drone.replan(obstacles)
        replanning_elapsed = time.perf_counter() - started

        replan_success = updated_path is not None

        if updated_path is not None:
            updated_cost = calculate_path_cost(updated_path)
            list(drone.run())
        else:
            updated_cost = float("nan")

        scenario_names.append(name)
        initial_costs.append(initial_cost)
        updated_costs.append(updated_cost)
        initial_times.append(initial_elapsed)
        replanning_times.append(replanning_elapsed)

        final_positions.append(drone.position)
        replanning_results.append(replan_success)

        print(f"Initial path cost: {initial_cost:.3f}")
        print(f"Updated path cost: {updated_cost:.3f}" if replan_success else "Updated path: Not found")
        print(f"Replanning successful: {replan_success}")
        print(f"Goal reached: {drone.reached_goal}")
        print(f"Safe stop: {drone.safe_stop}")

        visualize_navigation_comparison(
            grid=grid,
            start=start,
            goal=goal,
            original_path=original_path,
            updated_path=updated_path,
            trajectory=drone.visited_positions,
            obstacles=obstacles,
        )

    # Evaluate emergency scenarios separately.
    print("\n" + "=" * 60)
    print("EMERGENCY AND SAFETY EVALUATION")
    print("=" * 60)

    for scenario in emergency_scenarios:
        name = scenario["name"]
        obstacles = scenario["obstacles"]

        print(f"\nRunning emergency scenario: {name}")

        original_path = astar(grid, start, goal)

        if original_path is None:
            continue

        drone = DroneSimulator(grid, original_path)

        # Move to (2, 2) before introducing the emergency.
        drone.step()
        drone.step()

        updated_path = drone.replan(obstacles)

        # Emergency response is successful when the drone safely stops.
        emergency_success = (
            drone.emergency_detected
            and updated_path is None
            and drone.safe_stop
            and not drone.reached_goal
        )

        emergency_results.append(emergency_success)

        if drone.safe_stop:
            emergency_safe_stops += 1

        print(f"Drone position: {drone.position}")
        print(f"Emergency detected: {drone.emergency_detected}")
        print(f"Safe stop: {drone.safe_stop}")
        print(f"Safe stop reason: {drone.safe_stop_reason}")
        print(f"Emergency response successful: {emergency_success}")

        visualize_navigation_comparison(
            grid=grid,
            start=start,
            goal=goal,
            original_path=original_path,
            updated_path=updated_path,
            trajectory=drone.visited_positions,
            obstacles=obstacles,
        )

    # Evaluate unreachable-route safety scenarios separately.
    print("\n" + "=" * 60)
    print("UNREACHABLE ROUTE SAFETY EVALUATION")
    print("=" * 60)

    for scenario in unreachable_scenarios:
        name = scenario["name"]
        obstacles = scenario["obstacles"]

        print(f"\nRunning safety scenario: {name}")

        original_path = astar(grid, start, goal)

        if original_path is None:
            continue

        drone = DroneSimulator(grid, original_path)

        drone.step()
        drone.step()

        updated_path = drone.replan(obstacles)

        safety_success = (
            updated_path is None
            and drone.safe_stop
            and not drone.reached_goal
        )

        unreachable_results.append(safety_success)

        if drone.safe_stop:
            unreachable_safe_stops += 1

        print(f"Drone position: {drone.position}")
        print(f"Safe stop: {drone.safe_stop}")
        print(f"Safe stop reason: {drone.safe_stop_reason}")
        print(f"Unreachable-route handling successful: {safety_success}")

        visualize_navigation_comparison(
            grid=grid,
            start=start,
            goal=goal,
            original_path=original_path,
            updated_path=updated_path,
            trajectory=drone.visited_positions,
            obstacles=obstacles,
        )

    if not scenario_names:
        print("No navigation scenarios could be evaluated.")
        return

    # Calculate normal navigation metrics.
    goal_rate = calculate_goal_reaching_rate(
        final_positions,
        goal,
    )

    replanning_rate = calculate_success_rate(
        replanning_results,
    )

    emergency_response_rate = calculate_success_rate(
        emergency_results,
    )

    unreachable_route_rate = calculate_success_rate(
        unreachable_results,
    )

    print("\n" + "=" * 60)
    print("AGGREGATE PERFORMANCE")
    print("=" * 60)

    print(f"Goal-reaching rate: {goal_rate:.1f}%")
    print(f"Navigation replanning success rate: {replanning_rate:.1f}%")
    print(f"Emergency detection response rate: {emergency_response_rate:.1f}%")
    print(f"Unreachable-route handling rate: {unreachable_route_rate:.1f}%")

    print(
        f"Emergency safe stops: "
        f"{emergency_safe_stops}/{len(emergency_scenarios)}"
    )

    print(
        f"Unreachable-route safe stops: "
        f"{unreachable_safe_stops}/{len(unreachable_scenarios)}"
    )


    visualize_performance_dashboard(
        scenario_names=scenario_names,
        initial_path_costs=initial_costs,
        updated_path_costs=updated_costs,
        initial_planning_times=initial_times,
        replanning_times=replanning_times,
        goal_reaching_rate=goal_rate,
        replanning_success_rate=replanning_rate,
        emergency_safe_stop_count=emergency_safe_stops,
        emergency_total_scenarios=len(emergency_scenarios),
        unreachable_safe_stop_count=unreachable_safe_stops,
        unreachable_total_scenarios=len(unreachable_scenarios),
        emergency_response_rate=emergency_response_rate,
        unreachable_route_rate=unreachable_route_rate,
    )


if __name__ == "__main__":
    main()

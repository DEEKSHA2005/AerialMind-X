"""Reproducible benchmark scenario generation for AerialMind-X."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

from aerialmind.control.drone_simulator import DroneSimulator
from aerialmind.evaluation.metrics import calculate_path_cost
from aerialmind.planning.astar import Coordinate, astar


@dataclass(frozen=True)
class BenchmarkScenario:
    """Represent a navigation benchmark scenario."""

    name: str
    grid: list[list[int]]
    start: Coordinate
    goal: Coordinate
    obstacles: frozenset[Coordinate]
    obstacle_density: float
    seed: int


def generate_benchmark_scenario(
    name: str,
    rows: int = 30,
    cols: int = 30,
    obstacle_density: float = 0.0,
    seed: int = 42,
) -> BenchmarkScenario:
    """Generate a reproducible grid with a specified obstacle density."""
    if not name.strip():
        raise ValueError("Scenario name must not be empty.")

    if rows < 2 or cols < 2:
        raise ValueError("Grid dimensions must be at least 2.")

    if not 0.0 <= obstacle_density < 1.0:
        raise ValueError("Obstacle density must be between 0 and 1.")

    start = (0, 0)
    goal = (rows - 1, cols - 1)

    available_cells = [
        (row, col)
        for row in range(rows)
        for col in range(cols)
        if (row, col) not in {start, goal}
    ]

    obstacle_count = round(len(available_cells) * obstacle_density)

    generator = random.Random(seed)
    obstacles = frozenset(
        generator.sample(available_cells, obstacle_count)
    )

    grid = [[0 for _ in range(cols)] for _ in range(rows)]

    for row, col in obstacles:
        grid[row][col] = 1

    return BenchmarkScenario(
        name=name,
        grid=grid,
        start=start,
        goal=goal,
        obstacles=obstacles,
        obstacle_density=obstacle_density,
        seed=seed,
    )


def generate_standard_benchmark_scenarios(
    rows: int = 30,
    cols: int = 30,
    seed: int = 42,
) -> list[BenchmarkScenario]:
    """Generate open, low, medium, and high-density scenarios."""
    densities = [
        ("Open Environment", 0.0),
        ("Low Obstacle Density", 0.10),
        ("Medium Obstacle Density", 0.25),
        ("High Obstacle Density", 0.40),
    ]

    return [
        generate_benchmark_scenario(
            name=name,
            rows=rows,
            cols=cols,
            obstacle_density=density,
            seed=seed + index,
        )
        for index, (name, density) in enumerate(densities)
    ]


@dataclass(frozen=True)
class BenchmarkResult:
    """Store the measured outcome of one benchmark scenario."""

    scenario_name: str
    obstacle_density: float
    route_found: bool
    mission_success: bool
    safe_stop: bool | None
    initial_path_cost: float | None
    actual_path_cost: float | None
    planning_time_seconds: float
    mission_time_seconds: float | None
    steps: int


def run_benchmark_scenario(
    scenario: BenchmarkScenario,
) -> BenchmarkResult:
    """Execute and measure one navigation benchmark scenario."""
    planning_start = time.perf_counter()

    initial_path = astar(
        scenario.grid,
        scenario.start,
        scenario.goal,
    )

    planning_time = time.perf_counter() - planning_start

    if initial_path is None:
        return BenchmarkResult(
            scenario_name=scenario.name,
            obstacle_density=scenario.obstacle_density,
            route_found=False,
            mission_success=False,
            safe_stop=None,
            initial_path_cost=None,
            actual_path_cost=None,
            planning_time_seconds=planning_time,
            mission_time_seconds=None,
            steps=0,
        )

    simulator = DroneSimulator(
        grid=scenario.grid,
        path=initial_path,
    )

    mission_start = time.perf_counter()
    list(simulator.run())
    mission_time = time.perf_counter() - mission_start

    actual_path = simulator.visited_positions

    return BenchmarkResult(
        scenario_name=scenario.name,
        obstacle_density=scenario.obstacle_density,
        route_found=True,
        mission_success=simulator.reached_goal,
        safe_stop=simulator.safe_stop,
        initial_path_cost=calculate_path_cost(initial_path),
        actual_path_cost=calculate_path_cost(actual_path),
        planning_time_seconds=planning_time,
        mission_time_seconds=mission_time,
        steps=len(actual_path) - 1,
    )


def run_benchmark_suite(
    scenarios: list[BenchmarkScenario],
) -> list[BenchmarkResult]:
    """Execute a collection of benchmark scenarios."""
    if not scenarios:
        raise ValueError("At least one benchmark scenario is required.")

    return [
        run_benchmark_scenario(scenario)
        for scenario in scenarios
    ]


@dataclass(frozen=True)
class BenchmarkSummary:
    """Store aggregated benchmark performance."""

    total_scenarios: int
    route_found_count: int
    mission_success_count: int
    route_failure_count: int
    safe_stop_count: int
    success_rate: float
    average_planning_time: float
    average_mission_time: float | None
    average_initial_path_cost: float | None
    average_actual_path_cost: float | None


def summarize_benchmark_results(
    results: list[BenchmarkResult],
) -> BenchmarkSummary:
    """Aggregate measurements from a benchmark suite."""
    if not results:
        raise ValueError("At least one benchmark result is required.")

    route_found = [result for result in results if result.route_found]
    successful = [result for result in results if result.mission_success]

    mission_times = [
        result.mission_time_seconds
        for result in results
        if result.mission_time_seconds is not None
    ]

    initial_costs = [
        result.initial_path_cost
        for result in route_found
        if result.initial_path_cost is not None
    ]

    actual_costs = [
        result.actual_path_cost
        for result in route_found
        if result.actual_path_cost is not None
    ]

    return BenchmarkSummary(
        total_scenarios=len(results),
        route_found_count=len(route_found),
        mission_success_count=len(successful),
        route_failure_count=len(results) - len(route_found),
        safe_stop_count=sum(result.safe_stop is True for result in results),
        success_rate=len(successful) / len(results) * 100.0,
        average_planning_time=sum(
            result.planning_time_seconds for result in results
        ) / len(results),
        average_mission_time=(
            sum(mission_times) / len(mission_times)
            if mission_times else None
        ),
        average_initial_path_cost=(
            sum(initial_costs) / len(initial_costs)
            if initial_costs else None
        ),
        average_actual_path_cost=(
            sum(actual_costs) / len(actual_costs)
            if actual_costs else None
        ),
    )
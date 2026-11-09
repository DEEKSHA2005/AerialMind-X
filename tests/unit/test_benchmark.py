import pytest

from aerialmind.evaluation.benchmark import (
    BenchmarkScenario,
    generate_benchmark_scenario,
    generate_standard_benchmark_scenarios,
    run_benchmark_scenario,
    run_benchmark_suite,
    summarize_benchmark_results,
)


def test_generates_scenario_with_requested_dimensions():
    scenario = generate_benchmark_scenario(
        name="Test Scenario",
        rows=10,
        cols=12,
        obstacle_density=0.2,
    )

    assert len(scenario.grid) == 10
    assert all(len(row) == 12 for row in scenario.grid)


def test_preserves_start_and_goal_cells():
    scenario = generate_benchmark_scenario(
        name="Test Scenario",
        rows=10,
        cols=10,
        obstacle_density=0.4,
    )

    assert scenario.grid[0][0] == 0
    assert scenario.grid[9][9] == 0


def test_generates_reproducible_obstacles():
    first = generate_benchmark_scenario(
        name="Test",
        obstacle_density=0.2,
        seed=42,
    )
    second = generate_benchmark_scenario(
        name="Test",
        obstacle_density=0.2,
        seed=42,
    )

    assert first.obstacles == second.obstacles


def test_generates_requested_obstacle_density():
    scenario = generate_benchmark_scenario(
        name="Test",
        rows=10,
        cols=10,
        obstacle_density=0.2,
    )

    available_cells = 10 * 10 - 2
    expected_count = round(available_cells * 0.2)

    assert len(scenario.obstacles) == expected_count


def test_generates_four_standard_scenarios():
    scenarios = generate_standard_benchmark_scenarios()

    assert len(scenarios) == 4
    assert [scenario.obstacle_density for scenario in scenarios] == [
        0.0,
        0.10,
        0.25,
        0.40,
    ]


@pytest.mark.parametrize(
    "density",
    [-0.1, 1.0, 1.5],
)
def test_rejects_invalid_obstacle_density(density):
    with pytest.raises(ValueError, match="Obstacle density"):
        generate_benchmark_scenario(
            name="Invalid Scenario",
            obstacle_density=density,
        )


def test_rejects_invalid_grid_dimensions():
    with pytest.raises(ValueError, match="dimensions"):
        generate_benchmark_scenario(
            name="Invalid Scenario",
            rows=1,
            cols=10,
        )


def test_rejects_empty_scenario_name():
    with pytest.raises(ValueError, match="name"):
        generate_benchmark_scenario(
            name=" ",
        )


def test_runs_successful_benchmark_scenario():
    scenario = generate_benchmark_scenario(
        name="Open Test",
        rows=10,
        cols=10,
        obstacle_density=0.0,
    )

    result = run_benchmark_scenario(scenario)

    assert result.route_found is True
    assert result.mission_success is True
    assert result.safe_stop is False
    assert result.initial_path_cost > 0
    assert result.actual_path_cost > 0
    assert result.planning_time_seconds >= 0
    assert result.mission_time_seconds >= 0
    assert result.steps > 0


def test_records_unreachable_benchmark_scenario():
    scenario = BenchmarkScenario(
        name="Unreachable Test",
        grid=[
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
        ],
        start=(0, 0),
        goal=(0, 2),
        obstacles=frozenset({(0, 1), (1, 1), (2, 1)}),
        obstacle_density=0.33,
        seed=42,
    )

    result = run_benchmark_scenario(scenario)

    assert result.route_found is False
    assert result.mission_success is False
    assert result.safe_stop is None
    assert result.initial_path_cost is None
    assert result.actual_path_cost is None
    assert result.mission_time_seconds is None
    assert result.steps == 0


def test_runs_complete_benchmark_suite():
    scenarios = generate_standard_benchmark_scenarios(
        rows=10,
        cols=10,
    )

    results = run_benchmark_suite(scenarios)

    assert len(results) == 4
    assert all(result.planning_time_seconds >= 0 for result in results)


def test_rejects_empty_benchmark_suite():
    with pytest.raises(ValueError, match="At least one"):
        run_benchmark_suite([])


def test_summarizes_benchmark_results():
    scenarios = generate_standard_benchmark_scenarios(
        rows=10,
        cols=10,
    )
    results = run_benchmark_suite(scenarios)

    summary = summarize_benchmark_results(results)

    assert summary.total_scenarios == 4
    assert summary.route_found_count + summary.route_failure_count == 4
    assert 0 <= summary.success_rate <= 100
    assert summary.average_planning_time >= 0


def test_rejects_empty_benchmark_results():
    with pytest.raises(ValueError, match="At least one"):
        summarize_benchmark_results([])
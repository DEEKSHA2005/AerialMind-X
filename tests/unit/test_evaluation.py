import math

import pytest

from aerialmind.evaluation.metrics import (
    calculate_average_planning_time,
    calculate_goal_reaching_rate,
    calculate_path_cost,
    calculate_success_rate,
)


def test_calculates_straight_path_cost():
    path = [(0, 0), (0, 1), (0, 2)]

    assert calculate_path_cost(path) == pytest.approx(2.0)


def test_calculates_diagonal_path_cost():
    path = [(0, 0), (1, 1), (2, 2)]

    assert calculate_path_cost(path) == pytest.approx(2 * math.sqrt(2))


def test_calculates_mixed_path_cost():
    path = [(0, 0), (0, 1), (1, 2)]

    assert calculate_path_cost(path) == pytest.approx(1 + math.sqrt(2))


def test_rejects_empty_path():
    with pytest.raises(ValueError, match="must not be empty"):
        calculate_path_cost([])


def test_rejects_non_adjacent_path_positions():
    with pytest.raises(ValueError, match="must be adjacent"):
        calculate_path_cost([(0, 0), (2, 2)])


def test_calculates_success_rate():
    assert calculate_success_rate([True, True, False, True]) == pytest.approx(
        75.0
    )


def test_rejects_empty_success_results():
    with pytest.raises(ValueError, match="At least one"):
        calculate_success_rate([])


def test_calculates_goal_reaching_rate():
    goal = (7, 7)
    final_positions = [(7, 7), (6, 7), (7, 7), (0, 0)]

    assert calculate_goal_reaching_rate(
        final_positions, goal
    ) == pytest.approx(50.0)


def test_calculates_average_planning_time():
    assert calculate_average_planning_time([0.1, 0.2, 0.3]) == pytest.approx(
        0.2
    )


def test_rejects_invalid_planning_times():
    with pytest.raises(ValueError, match="finite and non-negative"):
        calculate_average_planning_time([0.1, -0.2])

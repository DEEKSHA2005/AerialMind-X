"""Tests for dynamic obstacle replanning."""

import pytest

from aerialmind.planning.dynamic_replanner import DynamicReplanner


def test_initial_route_is_created() -> None:
    planner = DynamicReplanner(
        grid=[[0] * 5 for _ in range(5)],
        start=(0, 0),
        goal=(4, 4),
    )

    assert planner.path is not None
    assert planner.path[0] == (0, 0)
    assert planner.path[-1] == (4, 4)
    assert planner.safe_stop is False


def test_new_obstacle_triggers_replanning() -> None:
    planner = DynamicReplanner(
        grid=[[0] * 5 for _ in range(5)],
        start=(2, 0),
        goal=(2, 4),
    )

    new_path = planner.update(
        current_position=(2, 0),
        obstacles={(2, 1), (2, 2), (2, 3)},
    )

    assert new_path is not None
    assert new_path[0] == (2, 0)
    assert new_path[-1] == (2, 4)
    assert all(planner.grid[r][c] == 0 for r, c in new_path)
    assert planner.safe_stop is False


def test_removed_obstacle_becomes_free_again() -> None:
    planner = DynamicReplanner(
        grid=[[0] * 5 for _ in range(5)],
        start=(2, 0),
        goal=(2, 4),
    )

    planner.update((2, 0), {(2, 2)})
    planner.update((2, 0), set())

    assert planner.grid[2][2] == 0
    assert planner.path is not None


def test_no_route_causes_safe_stop() -> None:
    planner = DynamicReplanner(
        grid=[[0] * 5 for _ in range(5)],
        start=(2, 0),
        goal=(2, 4),
    )

    # Block the entire middle column so no route can cross it.
    new_path = planner.update(
        current_position=(2, 0),
        obstacles={(0, 2), (1, 2), (2, 2), (3, 2), (4, 2)},
    )

    assert new_path is None
    assert planner.path is None
    assert planner.safe_stop is True



def test_obstacle_on_current_position_causes_safe_stop() -> None:
    planner = DynamicReplanner(
        grid=[[0] * 5 for _ in range(5)],
        start=(2, 0),
        goal=(2, 4),
    )

    assert planner.update((2, 0), {(2, 0)}) is None
    assert planner.safe_stop is True


def test_out_of_bounds_obstacle_is_rejected() -> None:
    planner = DynamicReplanner(
        grid=[[0] * 5 for _ in range(5)],
        start=(0, 0),
        goal=(4, 4),
    )

    with pytest.raises(ValueError, match="outside the grid"):
        planner.update((0, 0), {(5, 5)})

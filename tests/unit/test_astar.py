
import pytest

from aerialmind.planning.astar import astar


def test_finds_route_on_empty_grid():
    grid = [[0] * 5 for _ in range(5)]

    path = astar(grid, (0, 0), (4, 4))

    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (4, 4)
    assert len(path) == 5


def test_routes_around_obstacle():
    grid = [
        [0, 1, 0],
        [0, 1, 0],
        [0, 0, 0],
    ]

    path = astar(grid, (0, 0), (0, 2))

    assert path is not None
    assert (0, 1) not in path
    assert path[0] == (0, 0)
    assert path[-1] == (0, 2)


def test_returns_none_when_goal_is_unreachable():
    grid = [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ]

    assert astar(grid, (0, 0), (2, 2)) is None


def test_start_equal_to_goal():
    assert astar([[0]], (0, 0), (0, 0)) == [(0, 0)]


def test_rejects_blocked_start():
    with pytest.raises(ValueError, match="Start cell is blocked"):
        astar([[1, 0]], (0, 0), (0, 1))


def test_rejects_out_of_bounds_coordinate():
    with pytest.raises(ValueError, match="outside the grid"):
        astar([[0, 0]], (0, 0), (0, 2))


def test_rejects_non_rectangular_grid():
    with pytest.raises(ValueError, match="rectangular"):
        astar([[0, 0], [0]], (0, 0), (1, 0))


def test_diagonal_movement_does_not_cut_corners():
    grid = [
        [0, 1],
        [1, 0],
    ]

    assert astar(grid, (0, 0), (1, 1)) is None


def test_diagonal_path_is_allowed_when_safe():
    grid = [[0] * 3 for _ in range(3)]

    path = astar(grid, (0, 0), (2, 2))

    assert path == [(0, 0), (1, 1), (2, 2)]

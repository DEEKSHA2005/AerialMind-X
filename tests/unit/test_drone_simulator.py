
import pytest

from aerialmind.control.drone_simulator import DroneSimulator


def test_drone_reaches_goal():
    grid = [[0] * 3 for _ in range(3)]
    path = [(0, 0), (1, 1), (2, 2)]

    drone = DroneSimulator(grid, path)

    positions = list(drone.run())

    assert positions == [(1, 1), (2, 2)]
    assert drone.position == (2, 2)
    assert drone.reached_goal


def test_drone_records_all_visited_positions():
    grid = [[0] * 3 for _ in range(3)]
    path = [(0, 0), (0, 1), (1, 2)]

    drone = DroneSimulator(grid, path)
    list(drone.run())

    assert drone.visited_positions == path


def test_step_does_not_move_past_goal():
    drone = DroneSimulator([[0]], [(0, 0)])

    assert drone.step() == (0, 0)
    assert drone.visited_positions == [(0, 0)]


def test_rejects_empty_path():
    with pytest.raises(ValueError, match="Path must not be empty"):
        DroneSimulator([[0]], [])


def test_rejects_path_through_obstacle():
    grid = [
        [0, 1],
        [0, 0],
    ]
    path = [(0, 0), (0, 1), (1, 1)]

    with pytest.raises(ValueError, match="blocked cells"):
        DroneSimulator(grid, path)


def test_rejects_non_adjacent_positions():
    grid = [[0] * 4 for _ in range(4)]
    path = [(0, 0), (2, 2)]

    with pytest.raises(ValueError, match="must be adjacent"):
        DroneSimulator(grid, path)


def test_rejects_diagonal_corner_cutting():
    grid = [
        [0, 1],
        [1, 0],
    ]

    with pytest.raises(ValueError, match="blocked corners"):
        DroneSimulator(grid, [(0, 0), (1, 1)])

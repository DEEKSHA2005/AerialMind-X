"""Integration tests for drone movement and dynamic replanning."""

from aerialmind.control.drone_simulator import DroneSimulator


def test_drone_replans_around_new_obstacles() -> None:
    grid = [[0] * 5 for _ in range(5)]
    drone = DroneSimulator(
        grid,
        [(2, 0), (2, 1), (2, 2), (2, 3), (2, 4)],
    )

    drone.step()
    assert drone.position == (2, 1)

    new_path = drone.replan({(2, 2)})

    assert new_path is not None
    assert new_path[0] == drone.position
    assert new_path[-1] == drone.goal
    assert (2, 2) not in new_path

    list(drone.run())

    assert drone.reached_goal
    assert drone.position == (2, 4)
    assert (2, 2) not in drone.visited_positions


def test_drone_stops_when_no_safe_route_exists() -> None:
    grid = [[0] * 5 for _ in range(5)]
    drone = DroneSimulator(
        grid,
        [(2, 0), (2, 1), (2, 2), (2, 3), (2, 4)],
    )

    new_path = drone.replan(
        {(0, 2), (1, 2), (2, 2), (3, 2), (4, 2)}
    )

    assert new_path is None
    assert drone.safe_stop is True

    original_position = drone.position
    assert drone.step() == original_position
    assert list(drone.run()) == []
    assert drone.position == original_position


def test_removed_obstacle_allows_new_route() -> None:
    grid = [[0] * 5 for _ in range(5)]
    drone = DroneSimulator(
        grid,
        [(2, 0), (2, 1), (2, 2), (2, 3), (2, 4)],
    )

    drone.replan({(2, 2)})
    new_path = drone.replan(set())

    assert new_path is not None
    assert drone.safe_stop is False
    assert new_path[-1] == drone.goal


def test_drone_does_not_move_into_a_newly_blocked_next_cell() -> None:
    grid = [[0] * 5 for _ in range(5)]
    drone = DroneSimulator(
        grid,
        [(2, 0), (2, 1), (2, 2), (2, 3), (2, 4)],
    )

    drone.grid[2][1] = 1

    assert drone.step() == (2, 0)
    assert drone.safe_stop is True

from aerialmind.control.drone_simulator import DroneSimulator
from aerialmind.planning.astar import astar


def test_drone_replans_around_new_dynamic_obstacle():
    grid = [[0 for _ in range(8)] for _ in range(8)]
    start = (0, 0)
    goal = (7, 7)

    initial_path = astar(grid, start, goal)

    assert initial_path is not None

    drone = DroneSimulator(grid, initial_path)

    # Move two steps along the initial diagonal route.
    assert drone.step() == (1, 1)
    assert drone.step() == (2, 2)

    # Simulate a new obstacle on the next planned cell.
    new_obstacle = (3, 3)
    updated_path = drone.replan({new_obstacle})

    assert updated_path is not None
    assert updated_path[0] == drone.position
    assert updated_path[-1] == goal
    assert new_obstacle not in updated_path[1:]
    assert updated_path != [(2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7)]

    # The drone must reach the goal without crossing the obstacle.
    list(drone.run())

    assert drone.reached_goal
    assert not drone.safe_stop
    assert new_obstacle not in drone.visited_positions

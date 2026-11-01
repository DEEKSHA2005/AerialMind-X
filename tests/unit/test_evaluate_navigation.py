from aerialmind.control.drone_simulator import DroneSimulator
from aerialmind.planning.astar import astar


def test_evaluation_scenario_can_reach_goal_after_replanning():
    grid = [[0] * 8 for _ in range(8)]
    path = astar(grid, (0, 0), (7, 7))

    assert path is not None

    drone = DroneSimulator(grid, path)
    drone.step()
    drone.step()

    updated_path = drone.replan({(3, 3)})

    assert updated_path is not None
    assert (3, 3) not in updated_path[1:]

    list(drone.run())

    assert drone.reached_goal
    assert not drone.safe_stop


def test_evaluation_scenario_safe_stops_when_route_is_blocked():
    grid = [[0] * 8 for _ in range(8)]
    path = astar(grid, (0, 0), (7, 7))

    assert path is not None

    drone = DroneSimulator(grid, path)
    drone.step()
    drone.step()

    # A complete vertical barrier blocks every route to the goal.
    barrier = {(row, 3) for row in range(8)}
    updated_path = drone.replan(barrier)

    assert updated_path is None
    assert drone.safe_stop
    assert not drone.reached_goal

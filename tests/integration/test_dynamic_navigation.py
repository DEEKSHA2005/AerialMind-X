import numpy as np

from aerialmind.control.drone_simulator import DroneSimulator
from aerialmind.core.video_navigation_agent import VideoNavigationAgent
from aerialmind.perception.obstacle_detector import Detection
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


def test_drone_activates_safe_stop_when_current_position_is_blocked():
    grid = [[0 for _ in range(8)] for _ in range(8)]
    start = (0, 0)
    goal = (7, 7)

    initial_path = astar(grid, start, goal)
    assert initial_path is not None

    drone = DroneSimulator(grid, initial_path)

    drone.step()
    current_position = drone.position

    drone.replan({current_position})

    assert drone.emergency_detected
    assert drone.safe_stop
    assert drone.safe_stop_reason is not None
    assert drone.step() == current_position
    assert not drone.reached_goal


def test_drone_activates_safe_stop_when_no_route_exists():
    grid = [[0 for _ in range(8)] for _ in range(8)]
    start = (0, 0)
    goal = (7, 7)

    initial_path = astar(grid, start, goal)
    assert initial_path is not None

    drone = DroneSimulator(grid, initial_path)

    # Create a complete barrier between start and goal.
    obstacles = {(4, col) for col in range(8)}

    updated_path = drone.replan(obstacles)

    assert updated_path is None
    assert drone.safe_stop
    assert not drone.emergency_detected
    assert drone.safe_stop_reason == "No safe route available"
    assert drone.position == start
    assert not drone.reached_goal


def test_video_agent_replans_when_detected_obstacle_changes(
    monkeypatch,
):
    agent = VideoNavigationAgent(
        confidence_threshold=0.40,
        grid_rows=8,
        grid_cols=8,
    )

    agent.initialize_mission(start=(0, 0), goal=(7, 7))

    # Simulate YOLO detecting an obstacle in the next route cell.
    detection = Detection(
        label="car",
        confidence=0.95,
        bounding_box=(30, 30, 50, 50),
    )

    monkeypatch.setattr(
        agent.detector,
        "detect",
        lambda frame: [detection],
    )

    frame = np.zeros((80, 80, 3), dtype=np.uint8)

    result = agent.process_frame(frame, frame_index=1)

    assert result.replanned
    assert result.obstacles
    assert result.planned_path is not None
    assert result.planned_path[0] == (0, 0)
    assert result.drone_position == (1, 1)
    assert (3, 3) not in result.planned_path[1:]
    assert not result.safe_stop


def test_video_agent_activates_safe_stop_when_current_position_is_blocked(
    monkeypatch,
):
    agent = VideoNavigationAgent(grid_rows=8, grid_cols=8)
    agent.initialize_mission(start=(0, 0), goal=(7, 7))

    detections = [
        Detection("car", 0.95, (1, 1, 8, 8))
    ]

    monkeypatch.setattr(
        agent.detector,
        "detect",
        lambda frame: detections,
    )

    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    result = agent.process_frame(frame, frame_index=1)

    assert result.replanned is True
    assert result.emergency_detected is True
    assert result.safe_stop is True
    assert result.drone_position == (0, 0)
    assert result.safe_stop_reason is not None
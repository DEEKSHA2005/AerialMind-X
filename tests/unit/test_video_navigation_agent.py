import numpy as np
import pytest

from aerialmind.core.video_navigation_agent import VideoNavigationAgent


class FakeDetector:
    def detect(self, frame):
        return []


@pytest.fixture
def agent():
    video_agent = VideoNavigationAgent.__new__(VideoNavigationAgent)
    video_agent.detector = FakeDetector()
    video_agent.grid_rows = 10
    video_agent.grid_cols = 10
    video_agent.confidence_threshold = 0.40
    video_agent.simulator = None
    video_agent.goal = None
    video_agent.frame_results = []
    video_agent.previous_obstacles = set()
    return video_agent


def test_mission_initialization(agent):
    agent.initialize_mission()

    assert agent.simulator is not None
    assert agent.goal == (9, 9)
    assert agent.simulator.position == (0, 0)


def test_process_frame_updates_drone_position(agent):
    agent.initialize_mission()

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = agent.process_frame(frame, frame_index=1)

    assert result.frame_index == 1
    assert result.drone_position == (1, 1)
    assert result.annotated_frame.shape == frame.shape


def test_process_frame_rejects_invalid_image(agent):
    agent.initialize_mission()

    with pytest.raises(ValueError):
        agent.process_frame(
            np.array([]),
            frame_index=1,
        )


def test_process_frame_rejects_invalid_frame_index(agent):
    agent.initialize_mission()

    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        agent.process_frame(frame, frame_index=0)


def test_process_sequence_rejects_missing_directory(agent):
    with pytest.raises(FileNotFoundError):
        list(agent.process_sequence("missing_sequence_directory"))


def test_process_frame_replans_when_obstacles_change(agent):
    agent.initialize_mission()

    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    agent.process_frame(frame, frame_index=1)

    agent.detector.detect = lambda image: [
        type(
            "FakeDetection",
            (),
            {
                "confidence": 0.95,
                "bounding_box": (45, 45, 55, 55),
                "label": "car",
            },
        )()
    ]

    second_result = agent.process_frame(frame, frame_index=2)

    assert second_result.obstacles
    assert second_result.planned_path is not None
    assert second_result.planned_path[1] == second_result.drone_position
    assert not (
        set(second_result.planned_path[1:])
        & second_result.obstacles
    )


def test_process_frame_triggers_emergency_safe_stop(agent):
    agent.initialize_mission()

    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    agent.process_frame(frame, frame_index=1)

    current_position = agent.simulator.position
    row, col = current_position

    # Map the detected object's centre to the drone's current grid cell.
    cell_width = 100 / agent.grid_cols
    cell_height = 100 / agent.grid_rows

    center_x = (col + 0.5) * cell_width
    center_y = (row + 0.5) * cell_height

    agent.detector.detect = lambda image: [
        type(
            "FakeDetection",
            (),
            {
                "confidence": 0.95,
                "bounding_box": (
                    center_x - 2,
                    center_y - 2,
                    center_x + 2,
                    center_y + 2,
                ),
                "label": "car",
            },
        )()
    ]

    result = agent.process_frame(frame, frame_index=2)

    assert result.emergency_detected
    assert result.safe_stop
    assert result.drone_position == current_position

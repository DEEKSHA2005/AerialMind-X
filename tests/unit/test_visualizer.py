
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from aerialmind.planning.visualizer import visualize_path


def test_visualizes_valid_path(monkeypatch):
    grid = [
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
    ]
    path = [(0, 0), (0, 1), (1, 2), (2, 2)]

    monkeypatch.setattr(plt, "show", lambda: None)

    visualize_path(grid, path, (0, 0), (2, 2))

    figure = plt.gcf()
    assert figure.axes[0].get_title() == "AerialMind-X: A* Path Planning"

    plt.close("all")


def test_rejects_empty_grid():
    with pytest.raises(ValueError, match="Grid must not be empty"):
        visualize_path([], None, (0, 0), (0, 0))


def test_rejects_non_rectangular_grid():
    with pytest.raises(ValueError, match="rectangular"):
        visualize_path([[0, 0], [0]], None, (0, 0), (1, 0))


def test_rejects_blocked_start():
    with pytest.raises(ValueError, match="Start cell is blocked"):
        visualize_path([[1, 0]], None, (0, 0), (0, 1))


def test_rejects_path_through_blocked_cell():
    grid = [
        [0, 1],
        [0, 0],
    ]
    path = [(0, 0), (0, 1), (1, 1)]

    with pytest.raises(ValueError, match="blocked cells"):
        visualize_path(grid, path, (0, 0), (1, 1))


def test_rejects_path_with_wrong_endpoints():
    grid = [[0] * 3 for _ in range(3)]
    path = [(0, 1), (1, 2), (2, 2)]

    with pytest.raises(ValueError, match="begin at start"):
        visualize_path(grid, path, (0, 0), (2, 2))

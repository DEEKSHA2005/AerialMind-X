"""Visualization utilities for AerialMind-X evaluation results."""

from __future__ import annotations

import matplotlib.pyplot as plt

Coordinate = tuple[int, int]


def visualize_navigation_comparison(
    grid: list[list[int]],
    start: Coordinate,
    goal: Coordinate,
    original_path: list[Coordinate],
    updated_path: list[Coordinate] | None,
    trajectory: list[Coordinate],
    obstacles: set[Coordinate],
) -> None:
    """Visualize original route, replanned route, and drone trajectory."""
    if not grid or not grid[0]:
        raise ValueError("Grid must not be empty.")

    rows, cols = len(grid), len(grid[0])

    if any(len(row) != cols for row in grid):
        raise ValueError("Grid must be rectangular.")

    figure, axis = plt.subplots(figsize=(9, 8))

    axis.imshow(
        grid,
        cmap="Greys",
        origin="upper",
        vmin=0,
        vmax=1,
    )

    axis.set_xticks(range(cols))
    axis.set_yticks(range(rows))
    axis.set_xticks(
        [value - 0.5 for value in range(cols + 1)],
        minor=True,
    )
    axis.set_yticks(
        [value - 0.5 for value in range(rows + 1)],
        minor=True,
    )
    axis.grid(which="minor", color="lightgray", linewidth=1)
    axis.tick_params(which="minor", bottom=False, left=False)

    if original_path:
        axis.plot(
            [point[1] for point in original_path],
            [point[0] for point in original_path],
            color="dodgerblue",
            linestyle="--",
            linewidth=2,
            marker="o",
            markersize=4,
            label="Original route",
        )

    if updated_path:
        axis.plot(
            [point[1] for point in updated_path],
            [point[0] for point in updated_path],
            color="darkorange",
            linewidth=2.5,
            marker="o",
            markersize=4,
            label="Replanned route",
        )

    if trajectory:
        axis.plot(
            [point[1] for point in trajectory],
            [point[0] for point in trajectory],
            color="limegreen",
            linewidth=2,
            marker=".",
            markersize=8,
            label="Drone trajectory",
        )

    if obstacles:
        axis.scatter(
            [point[1] for point in obstacles],
            [point[0] for point in obstacles],
            color="red",
            marker="X",
            s=100,
            label="Dynamic obstacles",
        )

    axis.scatter(
        start[1],
        start[0],
        color="green",
        edgecolors="black",
        s=150,
        label="Start",
        zorder=5,
    )

    axis.scatter(
        goal[1],
        goal[0],
        color="purple",
        edgecolors="black",
        s=150,
        marker="*",
        label="Goal",
        zorder=5,
    )

    axis.set_title("AerialMind-X: Dynamic Navigation and Replanning")
    axis.set_xlabel("Column")
    axis.set_ylabel("Row")
    axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    axis.set_aspect("equal")

    figure.tight_layout()
    plt.show()




def visualize_performance_dashboard(
    scenario_names: list[str],
    initial_path_costs: list[float],
    updated_path_costs: list[float],
    initial_planning_times: list[float],
    replanning_times: list[float],
    goal_reaching_rate: float,
    replanning_success_rate: float,
    emergency_safe_stop_count: int,
    emergency_total_scenarios: int,
    unreachable_safe_stop_count: int,
    unreachable_total_scenarios: int,
    emergency_response_rate: float,
    unreachable_route_rate: float,
) -> None:
    """Display navigation, planning, and safety performance metrics."""
    if not scenario_names:
        raise ValueError("At least one scenario is required.")

    if not (
        len(scenario_names)
        == len(initial_path_costs)
        == len(updated_path_costs)
        == len(initial_planning_times)
        == len(replanning_times)
    ):
        raise ValueError("All scenario metric lists must have equal lengths.")

    if emergency_total_scenarios <= 0 or unreachable_total_scenarios <= 0:
        raise ValueError("Safety scenario counts must be greater than zero.")

    figure, axes = plt.subplots(2, 4, figsize=(24, 12))

    figure.suptitle(
        "AerialMind-X: Navigation Performance Dashboard",
        fontsize=20,
        fontweight="bold",
        y=0.98,
    )

    positions = list(range(len(scenario_names)))
    bar_width = 0.35

    # 1. Path cost comparison
    axis = axes[0, 0]

    initial_bars = axis.bar(
        [p - bar_width / 2 for p in positions],
        initial_path_costs,
        width=bar_width,
        label="Initial path",
    )

    updated_bars = axis.bar(
        [p + bar_width / 2 for p in positions],
        updated_path_costs,
        width=bar_width,
        label="Replanned path",
    )

    axis.set_title("Path Cost Comparison", fontweight="bold")
    axis.set_ylabel("Path cost")
    axis.set_xticks(positions)
    axis.set_xticklabels(scenario_names, rotation=20, ha="right")
    axis.legend(fontsize=8)
    axis.bar_label(initial_bars, fmt="%.2f", padding=3, fontsize=8)
    axis.bar_label(updated_bars, fmt="%.2f", padding=3, fontsize=8)
    axis.grid(axis="y", alpha=0.3)
    axis.set_axisbelow(True)

    # 2. Planning time comparison
    axis = axes[0, 1]

    initial_ms = [value * 1000 for value in initial_planning_times]
    replanning_ms = [value * 1000 for value in replanning_times]

    initial_bars = axis.bar(
        [p - bar_width / 2 for p in positions],
        initial_ms,
        width=bar_width,
        label="Initial planning",
    )

    replanning_bars = axis.bar(
        [p + bar_width / 2 for p in positions],
        replanning_ms,
        width=bar_width,
        label="Dynamic replanning",
    )

    axis.set_title("Planning Time Comparison", fontweight="bold")
    axis.set_ylabel("Time (milliseconds)")
    axis.set_xticks(positions)
    axis.set_xticklabels(scenario_names, rotation=20, ha="right")
    axis.legend(fontsize=8)
    axis.bar_label(initial_bars, fmt="%.2f", padding=3, fontsize=8)
    axis.bar_label(replanning_bars, fmt="%.2f", padding=3, fontsize=8)
    axis.grid(axis="y", alpha=0.3)
    axis.set_axisbelow(True)

    # 3. Goal-reaching performance
    axis = axes[0, 2]

    bars = axis.barh(["Goal reaching"], [goal_reaching_rate])
    axis.set_title("Goal-Reaching Performance", fontweight="bold")
    axis.set_xlabel("Success rate (%)")

    # Rotate the horizontal category label vertically
    axis.set_yticklabels(
        ["Goal reaching"],
        rotation=90,
        va="center",
        ha="center",
    )

    axis.set_xlim(0, 110)
    axis.bar_label(
        bars,
        labels=[f"{goal_reaching_rate:.1f}%"],
        padding=5,
        fontweight="bold",
    )
    axis.grid(axis="x", alpha=0.3)
    axis.set_axisbelow(True)

    # 4. Dynamic replanning performance
    axis = axes[0, 3]

    bars = axis.barh(["Replanning success"], [replanning_success_rate])
    axis.set_title("Dynamic Replanning Performance", fontweight="bold")
    axis.set_xlabel("Success rate (%)")

    axis.set_yticklabels(
        ["Replanning success"],
        rotation=90,
        va="center",
        ha="center",
    )

    axis.set_xlim(0, 110)
    axis.bar_label(
        bars,
        labels=[f"{replanning_success_rate:.1f}%"],
        padding=5,
        fontweight="bold",
    )
    axis.grid(axis="x", alpha=0.3)
    axis.set_axisbelow(True)

    # 5. Emergency response performance
    axis = axes[1, 0]

    bars = axis.barh(["Emergency response"], [emergency_response_rate])
    axis.set_title("Emergency Response Performance", fontweight="bold")
    axis.set_xlabel("Success rate (%)")

    axis.set_yticklabels(
        ["Emergency response"],
        rotation=90,
        va="center",
        ha="center",
    )

    axis.set_xlim(0, 110)
    axis.bar_label(
        bars,
        labels=[f"{emergency_response_rate:.1f}%"],
        padding=5,
        fontweight="bold",
    )
    axis.grid(axis="x", alpha=0.3)
    axis.set_axisbelow(True)

    # 6. Unreachable-route handling performance
    axis = axes[1, 1]

    bars = axis.barh(["Unreachable-route handling"], [unreachable_route_rate])
    axis.set_title("Unreachable-Route Safety", fontweight="bold")
    axis.set_xlabel("Success rate (%)")

    axis.set_yticklabels(
        ["Unreachable-route handling"],
        rotation=90,
        va="center",
        ha="center",
    ) 
    
    axis.set_xlim(0, 110)
    axis.bar_label(
        bars,
        labels=[f"{unreachable_route_rate:.1f}%"],
        padding=5,
        fontweight="bold",
    )
    axis.grid(axis="x", alpha=0.3)
    axis.set_axisbelow(True)

    # 7. Emergency safe-stop outcomes
    axis = axes[1, 2]

    emergency_failures = (
        emergency_total_scenarios - emergency_safe_stop_count
    )

    bars = axis.bar(
        ["Safe stops", "Other outcomes"],
        [emergency_safe_stop_count, emergency_failures],
    )

    axis.set_title("Emergency Safe-Stop Outcomes", fontweight="bold")
    axis.set_ylabel("Number of scenarios")
    axis.bar_label(bars, padding=4)
    axis.grid(axis="y", alpha=0.3)
    axis.set_axisbelow(True)

    # 8. Unreachable-route safe-stop outcomes
    axis = axes[1, 3]

    unreachable_failures = (
        unreachable_total_scenarios - unreachable_safe_stop_count
    )

    bars = axis.bar(
        ["Safe stops", "Other outcomes"],
        [unreachable_safe_stop_count, unreachable_failures],
    )

    axis.set_title("Unreachable-Route Safe Stops", fontweight="bold")
    axis.set_ylabel("Number of scenarios")
    axis.bar_label(bars, padding=4)
    axis.grid(axis="y", alpha=0.3)
    axis.set_axisbelow(True)

    # Improve overall dashboard appearance
    for axis in axes.flat:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(axis="both", labelsize=9)

    figure.subplots_adjust(
        left=0.06,
        right=0.98,
        top=0.91,
        bottom=0.13,
        wspace=0.45,
        hspace=0.65,
    )

    plt.show()

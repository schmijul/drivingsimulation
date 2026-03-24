import numpy as np

from drivesim.autonomy.planner import AStarPlanner


def test_astar_finds_path() -> None:
    grid = np.zeros((20, 20), dtype=np.float32)
    grid[8:12, 9:11] = 1.0
    planner = AStarPlanner(occupancy_threshold=0.6)
    path = planner.plan(grid, (1, 1), (18, 18))
    assert path
    assert path[0] == (1, 1)
    assert path[-1] == (18, 18)


def test_path_smoothing_keeps_turns_and_endpoints() -> None:
    planner = AStarPlanner()
    raw = [(0, 0), (0, 1), (0, 2), (1, 3), (2, 4)]
    smoothed = planner._smooth_path(raw)
    assert smoothed[0] == raw[0]
    assert smoothed[-1] == raw[-1]
    assert len(smoothed) < len(raw)

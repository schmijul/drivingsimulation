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

from __future__ import annotations

import heapq
import math
from typing import Dict, List, Tuple

import numpy as np

VecI = Tuple[int, int]


class AStarPlanner:
    def __init__(self, occupancy_threshold: float = 0.6, obstacle_cost_scale: float = 4.0, smooth_path: bool = True):
        self.occupancy_threshold = occupancy_threshold
        self.obstacle_cost_scale = obstacle_cost_scale
        self.smooth_path_enabled = smooth_path

    def _neighbors(self, node: VecI, shape: tuple[int, int]) -> List[Tuple[VecI, float]]:
        y, x = node
        cands = [
            ((y - 1, x), 1.0),
            ((y + 1, x), 1.0),
            ((y, x - 1), 1.0),
            ((y, x + 1), 1.0),
            ((y - 1, x - 1), math.sqrt(2.0)),
            ((y + 1, x + 1), math.sqrt(2.0)),
            ((y - 1, x + 1), math.sqrt(2.0)),
            ((y + 1, x - 1), math.sqrt(2.0)),
        ]
        return [c for c in cands if 0 <= c[0][0] < shape[0] and 0 <= c[0][1] < shape[1]]

    @staticmethod
    def _h(a: VecI, b: VecI) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _cell_cost(self, grid: np.ndarray, node: VecI) -> float:
        return float(max(0.0, grid[node[0], node[1]]) * self.obstacle_cost_scale)

    @staticmethod
    def _smooth_path(path: List[VecI]) -> List[VecI]:
        if len(path) <= 2:
            return path
        smoothed = [path[0]]
        prev_dy = path[1][0] - path[0][0]
        prev_dx = path[1][1] - path[0][1]
        for i in range(1, len(path) - 1):
            next_dy = path[i + 1][0] - path[i][0]
            next_dx = path[i + 1][1] - path[i][1]
            if (next_dy, next_dx) != (prev_dy, prev_dx):
                smoothed.append(path[i])
            prev_dy, prev_dx = next_dy, next_dx
        smoothed.append(path[-1])
        return smoothed

    def plan(self, grid: np.ndarray, start: VecI, goal: VecI) -> List[VecI]:
        if grid[goal[0], goal[1]] >= self.occupancy_threshold:
            return []

        open_heap: List[Tuple[float, VecI]] = [(0.0, start)]
        came_from: Dict[VecI, VecI] = {}
        g_score: Dict[VecI, float] = {start: 0.0}

        while open_heap:
            _, current = heapq.heappop(open_heap)
            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                if self.smooth_path_enabled:
                    return self._smooth_path(path)
                return path

            for nxt, move_cost in self._neighbors(current, grid.shape):
                if grid[nxt[0], nxt[1]] >= self.occupancy_threshold:
                    continue
                tentative_g = g_score[current] + move_cost + self._cell_cost(grid, nxt)
                if tentative_g < g_score.get(nxt, 1e18):
                    came_from[nxt] = current
                    g_score[nxt] = tentative_g
                    f = tentative_g + self._h(nxt, goal)
                    heapq.heappush(open_heap, (f, nxt))

        return []

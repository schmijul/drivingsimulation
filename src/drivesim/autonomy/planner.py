from __future__ import annotations

import heapq
from typing import Dict, List, Tuple

import numpy as np

VecI = Tuple[int, int]


class AStarPlanner:
    def __init__(self, occupancy_threshold: float = 0.6):
        self.occupancy_threshold = occupancy_threshold

    def _neighbors(self, node: VecI, shape: tuple[int, int]) -> List[VecI]:
        y, x = node
        cands = [
            (y - 1, x),
            (y + 1, x),
            (y, x - 1),
            (y, x + 1),
            (y - 1, x - 1),
            (y + 1, x + 1),
            (y - 1, x + 1),
            (y + 1, x - 1),
        ]
        return [c for c in cands if 0 <= c[0] < shape[0] and 0 <= c[1] < shape[1]]

    @staticmethod
    def _h(a: VecI, b: VecI) -> float:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

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
                return path

            for nxt in self._neighbors(current, grid.shape):
                if grid[nxt[0], nxt[1]] >= self.occupancy_threshold:
                    continue
                tentative_g = g_score[current] + self._h(current, nxt)
                if tentative_g < g_score.get(nxt, 1e18):
                    came_from[nxt] = current
                    g_score[nxt] = tentative_g
                    f = tentative_g + self._h(nxt, goal)
                    heapq.heappush(open_heap, (f, nxt))

        return []

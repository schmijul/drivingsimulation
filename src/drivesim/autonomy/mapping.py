from __future__ import annotations

import numpy as np

from drivesim.autonomy.sensors import LidarObservation
from drivesim.core.types import VehicleState, World


class OccupancyGridMapper:
    def __init__(self, world: World, resolution: float = 8.0):
        self.resolution = resolution
        self.cols = int(world.width // resolution) + 1
        self.rows = int(world.height // resolution) + 1
        self.grid = np.zeros((self.rows, self.cols), dtype=np.float32)

    def world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        gx = max(0, min(self.cols - 1, int(x / self.resolution)))
        gy = max(0, min(self.rows - 1, int(y / self.resolution)))
        return gy, gx

    def mark_free(self, x: float, y: float) -> None:
        gy, gx = self.world_to_grid(x, y)
        self.grid[gy, gx] = max(0.0, self.grid[gy, gx] - 0.25)

    def mark_occupied(self, x: float, y: float) -> None:
        gy, gx = self.world_to_grid(x, y)
        self.grid[gy, gx] = min(1.0, self.grid[gy, gx] + 0.45)

    def update(self, pose: VehicleState, obs: LidarObservation) -> np.ndarray:
        for angle, distance in zip(obs.angles, obs.distances):
            d = 0.0
            while d + 1.0 < distance:
                fx = pose.x + np.cos(angle) * d
                fy = pose.y + np.sin(angle) * d
                self.mark_free(float(fx), float(fy))
                d += self.resolution * 0.5

            ox = pose.x + np.cos(angle) * distance
            oy = pose.y + np.sin(angle) * distance
            self.mark_occupied(float(ox), float(oy))

        return self.grid

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

from drivesim.core.types import SimState


@dataclass
class LidarObservation:
    angles: List[float]
    distances: List[float]
    max_range: float


class LidarSensor:
    def __init__(self, rays: int = 41, fov: float = math.radians(180), max_range: float = 120.0, step: float = 2.0):
        self.rays = rays
        self.fov = fov
        self.max_range = max_range
        self.step = step

    @staticmethod
    def _occupied(x: float, y: float, state: SimState) -> bool:
        if x < 0 or y < 0 or x > state.world.width or y > state.world.height:
            return True
        for obs in state.world.obstacles:
            if obs.x <= x <= obs.x + obs.w and obs.y <= y <= obs.y + obs.h:
                return True
        for obs in state.world.dynamic_obstacles:
            if obs.x <= x <= obs.x + obs.w and obs.y <= y <= obs.y + obs.h:
                return True
        return False

    def read(self, state: SimState) -> LidarObservation:
        v = state.vehicle
        angles: List[float] = []
        distances: List[float] = []
        if self.rays <= 1:
            ray_offsets = [0.0]
        else:
            ray_offsets = [(-self.fov / 2.0) + i * (self.fov / (self.rays - 1)) for i in range(self.rays)]

        for offset in ray_offsets:
            angle = v.yaw + offset
            angles.append(angle)
            dist = 0.0
            while dist < self.max_range:
                px = v.x + math.cos(angle) * dist
                py = v.y + math.sin(angle) * dist
                if self._occupied(px, py, state):
                    break
                dist += self.step
            distances.append(min(dist, self.max_range))

        return LidarObservation(angles=angles, distances=distances, max_range=self.max_range)

from __future__ import annotations

import math
from typing import List, Tuple

from drivesim.core.types import Action, VehicleState

Vec2 = Tuple[float, float]


class PathController:
    def __init__(self, lookahead: int = 6):
        self.lookahead = lookahead

    def compute_action(self, state: VehicleState, path: List[Vec2]) -> Action:
        if not path:
            return Action(throttle=0.0, steering=0.0)

        idx = min(len(path) - 1, self.lookahead)
        tx, ty = path[idx]
        dx, dy = tx - state.x, ty - state.y
        desired = math.atan2(dy, dx)
        err = (desired - state.yaw + math.pi) % (2.0 * math.pi) - math.pi

        steering = max(-1.0, min(1.0, err / 0.8))
        distance = math.hypot(dx, dy)
        throttle = 0.75 if distance > 30 else 0.35
        if abs(err) > 1.1:
            throttle = 0.2

        return Action(throttle=throttle, steering=steering)

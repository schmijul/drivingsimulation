from __future__ import annotations

import math

from drivesim.core.types import Action, VehicleState

MAX_STEER = 0.55
MAX_SPEED = 85.0
WHEELBASE = 22.0
DRAG = 0.98
ACCEL = 58.0


def step_vehicle(state: VehicleState, action: Action, dt: float) -> VehicleState:
    steering = max(-1.0, min(1.0, action.steering)) * MAX_STEER
    throttle = max(-1.0, min(1.0, action.throttle))

    speed = state.speed + throttle * ACCEL * dt
    speed = max(-MAX_SPEED * 0.35, min(MAX_SPEED, speed))
    speed *= DRAG

    yaw_rate = 0.0
    if abs(steering) > 1e-5:
        yaw_rate = (speed / WHEELBASE) * math.tan(steering)

    yaw = state.yaw + yaw_rate * dt
    x = state.x + speed * math.cos(yaw) * dt
    y = state.y + speed * math.sin(yaw) * dt

    return VehicleState(x=x, y=y, yaw=yaw, speed=speed, steering=steering)

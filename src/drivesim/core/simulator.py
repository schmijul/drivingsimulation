from __future__ import annotations

from drivesim.core.types import Action, SimState, VehicleState, World
from drivesim.core.vehicle import step_vehicle

CAR_RADIUS = 9.0


def _inside_obstacle(x: float, y: float, world: World) -> bool:
    if x < 0 or y < 0 or x > world.width or y > world.height:
        return True
    for obs in world.obstacles:
        if obs.x <= x <= obs.x + obs.w and obs.y <= y <= obs.y + obs.h:
            return True
    for obs in world.dynamic_obstacles:
        if obs.x <= x <= obs.x + obs.w and obs.y <= y <= obs.y + obs.h:
            return True
    return False


def _collision(state: VehicleState, world: World) -> bool:
    offsets = [
        (0.0, 0.0),
        (CAR_RADIUS, 0.0),
        (-CAR_RADIUS, 0.0),
        (0.0, CAR_RADIUS),
        (0.0, -CAR_RADIUS),
    ]
    return any(_inside_obstacle(state.x + ox, state.y + oy, world) for ox, oy in offsets)


class Simulator:
    def __init__(self, world: World, dt: float = 0.05):
        self.world = world
        self.dt = dt
        self._state = self._initial_state()

    def _initial_state(self) -> SimState:
        start_x, start_y = self.world.start
        return SimState(
            vehicle=VehicleState(x=start_x, y=start_y, yaw=0.0),
            collided=False,
            t=0.0,
            world=self.world,
        )

    def reset(self) -> SimState:
        self._state = self._initial_state()
        return self._state

    def get_state(self) -> SimState:
        return self._state

    def _advance_dynamic_obstacles(self, dt: float) -> None:
        for obs in self.world.dynamic_obstacles:
            next_x = obs.x + obs.vx * dt
            next_y = obs.y + obs.vy * dt

            hit_x_wall = next_x < 0.0 or next_x + obs.w > self.world.width
            hit_y_wall = next_y < 0.0 or next_y + obs.h > self.world.height
            if hit_x_wall:
                obs.vx *= -1.0
                next_x = obs.x + obs.vx * dt
            if hit_y_wall:
                obs.vy *= -1.0
                next_y = obs.y + obs.vy * dt

            blocked = False
            for static_obs in self.world.obstacles:
                overlap = not (
                    next_x + obs.w < static_obs.x
                    or next_x > static_obs.x + static_obs.w
                    or next_y + obs.h < static_obs.y
                    or next_y > static_obs.y + static_obs.h
                )
                if overlap:
                    blocked = True
                    break

            if blocked:
                obs.vx *= -1.0
                obs.vy *= -1.0
            else:
                obs.x = max(0.0, min(next_x, self.world.width - obs.w))
                obs.y = max(0.0, min(next_y, self.world.height - obs.h))

    def step(self, action: Action) -> SimState:
        if self._state.collided:
            return self._state
        self._advance_dynamic_obstacles(self.dt)
        next_vehicle = step_vehicle(self._state.vehicle, action, self.dt)
        collided = _collision(next_vehicle, self.world)
        if collided:
            next_vehicle = self._state.vehicle
        self._state = SimState(
            vehicle=next_vehicle,
            collided=collided,
            t=self._state.t + self.dt,
            world=self.world,
            path=self._state.path,
        )
        return self._state

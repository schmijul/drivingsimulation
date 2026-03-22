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

    def step(self, action: Action) -> SimState:
        if self._state.collided:
            return self._state
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

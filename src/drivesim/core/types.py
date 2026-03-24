from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

Vec2 = Tuple[float, float]


@dataclass
class VehicleState:
    x: float
    y: float
    yaw: float
    speed: float = 0.0
    steering: float = 0.0


@dataclass
class Action:
    throttle: float
    steering: float


@dataclass
class Obstacle:
    x: float
    y: float
    w: float
    h: float


@dataclass
class DynamicObstacle(Obstacle):
    vx: float = 0.0
    vy: float = 0.0


@dataclass
class World:
    width: float
    height: float
    obstacles: List[Obstacle]
    start: Vec2
    goal: Vec2
    dynamic_obstacles: List[DynamicObstacle] = field(default_factory=list)


@dataclass
class SimState:
    vehicle: VehicleState
    collided: bool
    t: float
    world: World
    path: List[Vec2] = field(default_factory=list)

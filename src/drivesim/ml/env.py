from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from drivesim.autonomy.controller import PathController
from drivesim.autonomy.mapping import OccupancyGridMapper
from drivesim.autonomy.planner import AStarPlanner
from drivesim.autonomy.sensors import LidarSensor
from drivesim.core.scenario import default_world
from drivesim.core.simulator import Simulator
from drivesim.core.types import Action, SimState


@dataclass
class EnvConfig:
    max_steps: int = 1000


class DriveSimEnv:
    def __init__(self, config: EnvConfig | None = None):
        self.config = config or EnvConfig()
        self.world = default_world()
        self.sim = Simulator(self.world)
        self.lidar = LidarSensor()
        self.mapper = OccupancyGridMapper(self.world)
        self.planner = AStarPlanner()
        self.controller = PathController()
        self._steps = 0

    def reset(self, seed: int | None = None) -> Dict:
        del seed
        self._steps = 0
        state = self.sim.reset()
        self.mapper = OccupancyGridMapper(self.world)
        return self._observation(state)

    def _grid_goal(self) -> Tuple[int, int]:
        gy, gx = self.mapper.world_to_grid(self.world.goal[0], self.world.goal[1])
        return gy, gx

    def _grid_pos(self, state: SimState) -> Tuple[int, int]:
        gy, gx = self.mapper.world_to_grid(state.vehicle.x, state.vehicle.y)
        return gy, gx

    def _observation(self, state: SimState) -> Dict:
        obs = self.lidar.read(state)
        grid = self.mapper.update(state.vehicle, obs)
        front_slice = obs.distances[len(obs.distances) // 2 - 2 : len(obs.distances) // 2 + 3]
        return {
            "pose": np.array([state.vehicle.x, state.vehicle.y, state.vehicle.yaw, state.vehicle.speed], dtype=np.float32),
            "goal": np.array([self.world.goal[0], self.world.goal[1]], dtype=np.float32),
            "grid": grid.copy(),
            "lidar": np.array(obs.distances, dtype=np.float32),
            "lidar_front": [float(v) for v in front_slice],
            "collided": state.collided,
        }

    def _compute_path_world(self, grid: np.ndarray, state: SimState) -> list[tuple[float, float]]:
        start = self._grid_pos(state)
        goal = self._grid_goal()
        path_grid = self.planner.plan(grid, start, goal)
        world_path = [
            (gx * self.mapper.resolution + 0.5 * self.mapper.resolution, gy * self.mapper.resolution + 0.5 * self.mapper.resolution)
            for gy, gx in path_grid
        ]
        return world_path

    def autopilot_action(self, state: SimState) -> Action:
        obs = self._observation(state)
        path_world = self._compute_path_world(obs["grid"], state)
        self.sim.get_state().path = path_world
        return self.controller.compute_action(state.vehicle, path_world)

    def step(self, action: Action) -> Tuple[Dict, float, bool, Dict]:
        self._steps += 1
        state = self.sim.step(action)
        obs = self._observation(state)

        dx = state.vehicle.x - self.world.goal[0]
        dy = state.vehicle.y - self.world.goal[1]
        dist_goal = float((dx * dx + dy * dy) ** 0.5)
        done = state.collided or dist_goal < 18.0 or self._steps >= self.config.max_steps
        reward = 1.0 / (1.0 + 0.01 * dist_goal)
        if state.collided:
            reward -= 1.25
        if dist_goal < 18.0:
            reward += 4.0

        info = {"distance_to_goal": dist_goal, "steps": self._steps}
        return obs, reward, done, info

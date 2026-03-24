from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Tuple

import numpy as np

from drivesim.autonomy.controller import PathController
from drivesim.autonomy.mapping import OccupancyGridMapper
from drivesim.autonomy.planner import AStarPlanner
from drivesim.autonomy.sensors import LidarSensor
from drivesim.core.scenario import available_scenarios, build_world, generate_chunk_obstacles
from drivesim.core.simulator import Simulator
from drivesim.core.types import Action, DynamicObstacle, SimState


@dataclass
class EnvConfig:
    max_steps: int = 1000
    map_name: str = "default"
    auto_expand: bool = False
    chunk_size: int = 400
    expand_margin: float = 85.0
    dynamic_obstacle_count: int = 2
    dynamic_obstacle_speed: float = 52.0
    seed: int = 7


class DriveSimEnv:
    def __init__(self, config: EnvConfig | None = None):
        self.config = config or EnvConfig()
        self.map_name = self.config.map_name
        self.world = build_world(self.map_name)
        self.sim = Simulator(self.world)
        self.lidar = LidarSensor()
        self.mapper = OccupancyGridMapper(self.world)
        self.planner = AStarPlanner()
        self.controller = PathController()
        self._steps = 0
        self.seed = self.config.seed
        self._rng = np.random.default_rng(self.seed)
        self.auto_expand = self.config.auto_expand
        self.chunk_size = self.config.chunk_size
        self.expand_margin = self.config.expand_margin
        self.dynamic_obstacle_count = self.config.dynamic_obstacle_count
        self.dynamic_obstacle_speed = self.config.dynamic_obstacle_speed
        self._expansion_seed = sum((i + 1) * ord(c) for i, c in enumerate(self.map_name))
        self._generated_chunks: set[tuple[int, int]] = set()
        self._seed_existing_chunks()
        self._spawn_dynamic_obstacles()

    def available_maps(self) -> list[str]:
        return available_scenarios()

    def set_map(self, map_name: str) -> None:
        self.map_name = map_name
        self.world = build_world(map_name)
        self.sim = Simulator(self.world)
        self.mapper = OccupancyGridMapper(self.world)
        self._steps = 0
        self._expansion_seed = sum((i + 1) * ord(c) for i, c in enumerate(self.map_name))
        self._generated_chunks = set()
        self._seed_existing_chunks()
        self._spawn_dynamic_obstacles()

    def toggle_auto_expand(self) -> bool:
        self.auto_expand = not self.auto_expand
        return self.auto_expand

    def _point_is_free(self, x: float, y: float, margin: float = 14.0) -> bool:
        if x < margin or y < margin or x > self.world.width - margin or y > self.world.height - margin:
            return False
        for obs in self.world.obstacles:
            if obs.x - margin <= x <= obs.x + obs.w + margin and obs.y - margin <= y <= obs.y + obs.h + margin:
                return False
        for obs in self.world.dynamic_obstacles:
            if obs.x - margin <= x <= obs.x + obs.w + margin and obs.y - margin <= y <= obs.y + obs.h + margin:
                return False
        return True

    def _sample_free_point(self, rng: np.random.Generator, tries: int = 200) -> tuple[float, float]:
        for _ in range(tries):
            x = float(rng.uniform(20.0, self.world.width - 20.0))
            y = float(rng.uniform(20.0, self.world.height - 20.0))
            if self._point_is_free(x, y):
                return x, y
        return self.world.start

    def randomize_episode(
        self,
        rng: np.random.Generator | None = None,
        min_goal_distance: float = 220.0,
    ) -> Dict:
        rng = rng or self._rng
        start = self._sample_free_point(rng)
        goal = start
        for _ in range(120):
            cand = self._sample_free_point(rng)
            if math.hypot(cand[0] - start[0], cand[1] - start[1]) >= min_goal_distance:
                goal = cand
                break
            goal = cand

        self.world.start = start
        self.world.goal = goal
        self.sim = Simulator(self.world)
        self._steps = 0
        state = self.sim.get_state()
        return self._observation(state)

    def _chunk_counts(self) -> tuple[int, int]:
        cx = int(math.ceil(self.world.width / self.chunk_size))
        cy = int(math.ceil(self.world.height / self.chunk_size))
        return cx, cy

    def _seed_existing_chunks(self) -> None:
        chunk_x, chunk_y = self._chunk_counts()
        for cy in range(chunk_y):
            for cx in range(chunk_x):
                self._generated_chunks.add((cx, cy))

    @staticmethod
    def _overlap_rect(
        ax: float,
        ay: float,
        aw: float,
        ah: float,
        bx: float,
        by: float,
        bw: float,
        bh: float,
    ) -> bool:
        return not (ax + aw <= bx or ax >= bx + bw or ay + ah <= by or ay >= by + bh)

    def _spawn_dynamic_obstacles(self) -> None:
        self.world.dynamic_obstacles = []
        if self.dynamic_obstacle_count <= 0:
            return

        base = int(self._rng.integers(0, 2**31 - 1))
        seed = base + self._expansion_seed + int(self.world.width) * 11 + int(self.world.height) * 7
        rng = np.random.default_rng(seed)
        dynamic_obstacles = []

        for _ in range(self.dynamic_obstacle_count):
            placed = False
            for _ in range(250):
                w = float(rng.uniform(18.0, 34.0))
                h = float(rng.uniform(18.0, 34.0))
                x = float(rng.uniform(20.0, max(21.0, self.world.width - w - 20.0)))
                y = float(rng.uniform(20.0, max(21.0, self.world.height - h - 20.0)))

                blocked = False
                for obs in self.world.obstacles:
                    if self._overlap_rect(x, y, w, h, obs.x - 12.0, obs.y - 12.0, obs.w + 24.0, obs.h + 24.0):
                        blocked = True
                        break
                if blocked:
                    continue
                for obs in dynamic_obstacles:
                    if self._overlap_rect(
                        x,
                        y,
                        w,
                        h,
                        obs["x"] - 10.0,
                        obs["y"] - 10.0,
                        obs["w"] + 20.0,
                        obs["h"] + 20.0,
                    ):
                        blocked = True
                        break
                if blocked:
                    continue

                if math.hypot(x - self.world.start[0], y - self.world.start[1]) < 85.0:
                    continue
                if math.hypot(x - self.world.goal[0], y - self.world.goal[1]) < 70.0:
                    continue

                angle = float(rng.uniform(0.0, 2.0 * math.pi))
                speed = float(rng.uniform(self.dynamic_obstacle_speed * 0.6, self.dynamic_obstacle_speed))
                dynamic_obstacles.append(
                    {
                        "x": x,
                        "y": y,
                        "w": w,
                        "h": h,
                        "vx": math.cos(angle) * speed,
                        "vy": math.sin(angle) * speed,
                    }
                )
                placed = True
                break

            if not placed:
                break

        self.world.dynamic_obstacles = [DynamicObstacle(**obs) for obs in dynamic_obstacles]

    def _expand_world_to(self, width: float, height: float) -> None:
        old_chunk_x, old_chunk_y = self._chunk_counts()
        self.world.width = max(self.world.width, width)
        self.world.height = max(self.world.height, height)
        self.mapper.ensure_world_size(self.world.width, self.world.height)
        new_chunk_x, new_chunk_y = self._chunk_counts()
        for cy in range(new_chunk_y):
            for cx in range(new_chunk_x):
                if (cx, cy) in self._generated_chunks:
                    continue
                self.world.obstacles.extend(
                    generate_chunk_obstacles(
                        chunk_x=cx,
                        chunk_y=cy,
                        chunk_size=self.chunk_size,
                        seed=self._expansion_seed,
                    )
                )
                self._generated_chunks.add((cx, cy))

        if new_chunk_x > old_chunk_x or new_chunk_y > old_chunk_y:
            self.world.goal = (self.world.width - 70.0, self.world.height - 70.0)

    def _maybe_expand_world(self, state: SimState) -> None:
        if not self.auto_expand:
            return
        v = state.vehicle
        grow_width = self.world.width
        grow_height = self.world.height
        if v.x > self.world.width - self.expand_margin:
            grow_width += self.chunk_size
        if v.y > self.world.height - self.expand_margin:
            grow_height += self.chunk_size
        if grow_width > self.world.width or grow_height > self.world.height:
            self._expand_world_to(grow_width, grow_height)

    def reset(self, seed: int | None = None) -> Dict:
        if seed is not None:
            self.seed = int(seed)
            self._rng = np.random.default_rng(self.seed)
        self._steps = 0
        self.world = build_world(self.map_name)
        self.sim = Simulator(self.world)
        self.mapper = OccupancyGridMapper(self.world)
        self._generated_chunks = set()
        self._seed_existing_chunks()
        self._spawn_dynamic_obstacles()
        state = self.sim.get_state()
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
        self._maybe_expand_world(self.sim.get_state())
        state = self.sim.step(action)
        self._maybe_expand_world(state)
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

        info = {
            "distance_to_goal": dist_goal,
            "steps": self._steps,
            "world_size": (self.world.width, self.world.height),
            "auto_expand": self.auto_expand,
            "dynamic_obstacles": len(self.world.dynamic_obstacles),
        }
        return obs, reward, done, info

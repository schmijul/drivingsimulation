from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Tuple

import numpy as np

from drivesim.autonomy.controller import PathController
from drivesim.autonomy.mapping import OccupancyGridMapper
from drivesim.autonomy.planner import AStarPlanner
from drivesim.autonomy.sensors import LidarSensor
from drivesim.core.scenario import available_scenarios, build_world, generate_chunk_obstacles
from drivesim.core.simulator import Simulator
from drivesim.core.types import Action, DynamicObstacle, SimState

try:
    import gymnasium as _gym_mod
except ImportError:  # pragma: no cover - optional dependency
    try:
        import gym as _gym_mod
    except ImportError:  # pragma: no cover - optional dependency
        _gym_mod = None

if _gym_mod is not None:  # pragma: no branch - straightforward import guard
    _GymEnvBase = _gym_mod.Env
    _spaces = _gym_mod.spaces
else:  # pragma: no cover - used when gym/gymnasium is not installed
    _GymEnvBase = object
    _spaces = None

_VALID_MAPPING_MODES = {"ground_truth", "sensor_driven"}


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
    mapping_mode: str = "ground_truth"


class DriveSimEnv:
    def __init__(self, config: EnvConfig | None = None):
        self.config = config or EnvConfig()
        self.map_name = self.config.map_name if self.config.map_name in available_scenarios() else "default"
        self.config.map_name = self.map_name
        self.mapping_mode = self.config.mapping_mode
        self.world = build_world(self.map_name)
        self.sim = Simulator(self.world)
        self.lidar = LidarSensor()
        self.mapper = OccupancyGridMapper(self.world, mapping_mode=self.mapping_mode)
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
        resolved_map_name = map_name if map_name in available_scenarios() else "default"
        self.map_name = resolved_map_name
        self.config.map_name = resolved_map_name
        self.world = build_world(self.map_name)
        self.sim = Simulator(self.world)
        self.mapper = OccupancyGridMapper(self.world, mapping_mode=self.mapping_mode)
        self._steps = 0
        self._expansion_seed = sum((i + 1) * ord(c) for i, c in enumerate(self.map_name))
        self._generated_chunks = set()
        self._seed_existing_chunks()
        self._spawn_dynamic_obstacles()

    def set_mapping_mode(self, mapping_mode: str) -> None:
        if mapping_mode not in _VALID_MAPPING_MODES:
            raise ValueError(f"unsupported mapping_mode: {mapping_mode!r}")
        self.mapping_mode = mapping_mode
        self.config.mapping_mode = mapping_mode
        self.mapper = OccupancyGridMapper(self.world, mapping_mode=self.mapping_mode)

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

    def _sample_free_point(self, rng: np.random.Generator, tries: int = 200, margin: float = 14.0) -> tuple[float, float]:
        for _ in range(tries):
            x = float(rng.uniform(20.0, self.world.width - 20.0))
            y = float(rng.uniform(20.0, self.world.height - 20.0))
            if self._point_is_free(x, y, margin=margin):
                return x, y
        return self.world.start

    def _has_reachable_path(self, start: tuple[float, float], goal: tuple[float, float], min_nodes: int = 8) -> bool:
        sgy, sgx = self.mapper.world_to_grid(start[0], start[1])
        ggy, ggx = self.mapper.world_to_grid(goal[0], goal[1])
        path = self.planner.plan(self.mapper.grid, (sgy, sgx), (ggy, ggx))
        return len(path) >= min_nodes

    def randomize_episode(
        self,
        rng: np.random.Generator | None = None,
        min_goal_distance: float = 160.0,
    ) -> Dict:
        rng = rng or self._rng
        start = self._sample_free_point(rng, margin=22.0)
        goal = start
        for _ in range(120):
            cand = self._sample_free_point(rng, margin=22.0)
            if math.hypot(cand[0] - start[0], cand[1] - start[1]) >= min_goal_distance:
                if self._has_reachable_path(start, cand):
                    goal = cand
                    break
            goal = cand

        self.world.start = start
        self.world.goal = goal
        self._spawn_dynamic_obstacles(rng_override=rng)
        self.sim = Simulator(self.world)
        self._steps = 0
        state = self.sim.get_state()
        goal_heading = math.atan2(goal[1] - start[1], goal[0] - start[0])
        yaw_noise = float(rng.normal(0.0, 0.55))
        state.vehicle.yaw = float(((goal_heading + yaw_noise + math.pi) % (2.0 * math.pi)) - math.pi)
        state.vehicle.speed = 0.0
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

    def _spawn_dynamic_obstacles(self, rng_override: np.random.Generator | None = None) -> None:
        self.world.dynamic_obstacles = []
        if self.dynamic_obstacle_count <= 0:
            return

        base_rng = rng_override or self._rng
        base = int(base_rng.integers(0, 2**31 - 1))
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
        self.mapper = OccupancyGridMapper(self.world, mapping_mode=self.mapping_mode)
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
            "map_name": self.map_name,
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
            "mapping_mode": self.mapping_mode,
            "map_name": self.map_name,
            "collided": state.collided,
            "goal_reached": dist_goal < 18.0,
        }
        return obs, reward, done, info


class DriveSimGymEnv(_GymEnvBase):
    """Gym/Gymnasium-compatible wrapper around DriveSimEnv."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: EnvConfig | None = None,
        observation_mode: str = "dict",
        map_names: list[str] | None = None,
        randomize_on_reset: bool = False,
        min_goal_distance: float = 160.0,
    ):
        if _gym_mod is None or _spaces is None:
            raise ImportError(
                "DriveSimGymEnv requires 'gymnasium' or 'gym'. "
                "Install with: pip install -e .[rl]"
            )
        if observation_mode not in {"dict", "flat"}:
            raise ValueError(f"unsupported observation_mode: {observation_mode!r}")

        base_cfg = config or EnvConfig()
        if base_cfg.auto_expand:
            raise ValueError(
                "DriveSimGymEnv requires a fixed observation space. "
                "Set EnvConfig(auto_expand=False)."
            )

        self._seed = int(base_cfg.seed)
        self._rng = np.random.default_rng(self._seed)
        self.allowed_map_names = tuple(dict.fromkeys(map_names or [base_cfg.map_name]))
        if not self.allowed_map_names:
            self.allowed_map_names = ("default",)
        invalid = [map_name for map_name in self.allowed_map_names if map_name not in available_scenarios()]
        if invalid:
            raise ValueError(f"unsupported map_names: {invalid!r}")
        self._episode_map_names = list(self.allowed_map_names)
        self.randomize_on_reset = bool(randomize_on_reset)
        self.min_goal_distance = float(min_goal_distance)
        self.stage_name = ""
        self.base_env = DriveSimEnv(base_cfg)
        self.observation_mode = observation_mode
        self._target_world_width, self._target_world_height = self._max_world_size()
        self._target_grid_shape = self._grid_shape_for_world(self._target_world_width, self._target_world_height)
        self.action_space = _spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = self._build_observation_space()

    def _max_world_size(self) -> tuple[float, float]:
        max_width = 0.0
        max_height = 0.0
        for map_name in self.allowed_map_names:
            world = build_world(map_name)
            max_width = max(max_width, world.width)
            max_height = max(max_height, world.height)
        return max_width, max_height

    def _grid_shape_for_world(self, width: float, height: float) -> tuple[int, int]:
        resolution = self.base_env.mapper.resolution
        rows = int(height // resolution) + 1
        cols = int(width // resolution) + 1
        return rows, cols

    def _build_observation_space(self):
        grid_shape = self._target_grid_shape
        lidar_rays = self.base_env.lidar.rays

        dict_space = _spaces.Dict(
            {
                "pose": _spaces.Box(
                    low=np.array([0.0, 0.0, -math.pi, -200.0], dtype=np.float32),
                    high=np.array(
                        [self._target_world_width, self._target_world_height, math.pi, 200.0],
                        dtype=np.float32,
                    ),
                    dtype=np.float32,
                ),
                "goal": _spaces.Box(
                    low=np.array([0.0, 0.0], dtype=np.float32),
                    high=np.array([self._target_world_width, self._target_world_height], dtype=np.float32),
                    dtype=np.float32,
                ),
                "grid": _spaces.Box(low=0.0, high=1.0, shape=grid_shape, dtype=np.float32),
                "lidar": _spaces.Box(
                    low=0.0,
                    high=float(self.base_env.lidar.max_range),
                    shape=(lidar_rays,),
                    dtype=np.float32,
                ),
                "collided": _spaces.Discrete(2),
            }
        )
        if self.observation_mode == "dict":
            return dict_space

        flat_dim = int(4 + 2 + np.prod(grid_shape) + lidar_rays + 1)
        return _spaces.Box(low=-np.inf, high=np.inf, shape=(flat_dim,), dtype=np.float32)

    def _coerce_action(self, action: Any) -> Action:
        if isinstance(action, Action):
            throttle = float(action.throttle)
            steering = float(action.steering)
        elif isinstance(action, dict):
            throttle = float(action.get("throttle", 0.0))
            steering = float(action.get("steering", 0.0))
        else:
            arr = np.asarray(action, dtype=np.float32).reshape(-1)
            if arr.size < 2:
                raise ValueError("action must contain at least two values: [throttle, steering]")
            throttle = float(arr[0])
            steering = float(arr[1])
        return Action(
            throttle=float(np.clip(throttle, -1.0, 1.0)),
            steering=float(np.clip(steering, -1.0, 1.0)),
        )

    def _pad_grid(self, grid: np.ndarray) -> np.ndarray:
        target_rows, target_cols = self._target_grid_shape
        rows, cols = grid.shape
        if rows > target_rows or cols > target_cols:
            raise ValueError(
                "observed grid shape exceeds configured observation_space: "
                f"got {grid.shape}, expected <= {self._target_grid_shape}"
            )
        padded = np.zeros(self._target_grid_shape, dtype=np.float32)
        padded[:rows, :cols] = grid
        return padded

    def _format_observation(self, obs: Dict[str, Any]) -> Dict[str, Any] | np.ndarray:
        dict_obs = {
            "pose": np.asarray(obs["pose"], dtype=np.float32),
            "goal": np.asarray(obs["goal"], dtype=np.float32),
            "grid": self._pad_grid(np.asarray(obs["grid"], dtype=np.float32)),
            "lidar": np.asarray(obs["lidar"], dtype=np.float32),
            "collided": int(bool(obs["collided"])),
        }
        if self.observation_mode == "dict":
            return dict_obs

        flat_obs = np.concatenate(
            [
                dict_obs["pose"],
                dict_obs["goal"],
                dict_obs["grid"].ravel(),
                dict_obs["lidar"],
                np.array([dict_obs["collided"]], dtype=np.float32),
            ]
        )
        return flat_obs.astype(np.float32, copy=False)

    def set_episode_profile(
        self,
        *,
        map_names: list[str] | None = None,
        dynamic_obstacle_count: int | None = None,
        mapping_mode: str | None = None,
        min_goal_distance: float | None = None,
        randomize_on_reset: bool | None = None,
        stage_name: str | None = None,
    ) -> None:
        if map_names is not None:
            unique_maps = list(dict.fromkeys(map_names))
            invalid = [name for name in unique_maps if name not in self.allowed_map_names]
            if invalid:
                raise ValueError(f"maps are outside allowed observation_space set: {invalid!r}")
            self._episode_map_names = unique_maps or list(self.allowed_map_names)
        if dynamic_obstacle_count is not None:
            value = int(dynamic_obstacle_count)
            self.base_env.dynamic_obstacle_count = value
            self.base_env.config.dynamic_obstacle_count = value
        if mapping_mode is not None:
            self.base_env.set_mapping_mode(mapping_mode)
        if min_goal_distance is not None:
            self.min_goal_distance = float(min_goal_distance)
        if randomize_on_reset is not None:
            self.randomize_on_reset = bool(randomize_on_reset)
        if stage_name is not None:
            self.stage_name = stage_name

    def apply_curriculum_stage(self, stage: dict[str, Any]) -> None:
        self.set_episode_profile(
            map_names=[str(name) for name in list(stage.get("maps", []))],
            dynamic_obstacle_count=int(stage.get("dynamic_obstacle_count", self.base_env.dynamic_obstacle_count)),
            mapping_mode=str(stage.get("mapping_mode", self.base_env.mapping_mode)),
            min_goal_distance=float(stage.get("min_goal_distance", self.min_goal_distance)),
            randomize_on_reset=True,
            stage_name=str(stage.get("name", "")),
        )

    def _choose_map_name(self, explicit_map_name: str | None = None) -> str:
        if explicit_map_name:
            if explicit_map_name not in self.allowed_map_names:
                raise ValueError(f"map_name {explicit_map_name!r} is outside allowed map_names")
            return explicit_map_name
        if len(self._episode_map_names) == 1:
            return self._episode_map_names[0]
        idx = int(self._rng.integers(0, len(self._episode_map_names)))
        return self._episode_map_names[idx]

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self._seed = int(seed)
            self._rng = np.random.default_rng(self._seed)

        options = options or {}
        if "map_names" in options:
            self.set_episode_profile(map_names=[str(name) for name in list(options["map_names"])])
        if "dynamic_obstacle_count" in options:
            self.set_episode_profile(dynamic_obstacle_count=int(options["dynamic_obstacle_count"]))
        if "mapping_mode" in options:
            self.set_episode_profile(mapping_mode=str(options["mapping_mode"]))
        if "min_goal_distance" in options:
            self.set_episode_profile(min_goal_distance=float(options["min_goal_distance"]))
        if "randomize_episode" in options:
            self.set_episode_profile(randomize_on_reset=bool(options["randomize_episode"]))
        if "stage_name" in options:
            self.stage_name = str(options["stage_name"])

        map_name = self._choose_map_name(str(options["map_name"]) if "map_name" in options else None)
        if map_name != self.base_env.map_name:
            self.base_env.set_map(map_name)

        obs = self.base_env.reset(seed=seed)
        if self.randomize_on_reset:
            obs = self.base_env.randomize_episode(rng=self._rng, min_goal_distance=self.min_goal_distance)

        info = {
            "map_name": self.base_env.map_name,
            "world_size": (self.base_env.world.width, self.base_env.world.height),
            "mapping_mode": self.base_env.mapping_mode,
            "dynamic_obstacles": len(self.base_env.world.dynamic_obstacles),
            "curriculum_stage": self.stage_name,
        }
        return self._format_observation(obs), info

    def step(self, action: Any):
        obs, reward, done, info = self.base_env.step(self._coerce_action(action))
        terminated = bool(info["collided"] or info["goal_reached"])
        truncated = bool(done and not terminated)
        info = dict(info)
        info["curriculum_stage"] = self.stage_name
        return self._format_observation(obs), float(reward), terminated, truncated, info

    def render(self):  # pragma: no cover - no renderer in wrapper
        return None

    def close(self) -> None:
        return None

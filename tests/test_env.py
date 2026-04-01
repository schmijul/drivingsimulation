import importlib

import pytest
from drivesim.core.types import Action
from drivesim.ml.env import DriveSimEnv, DriveSimGymEnv, EnvConfig
import numpy as np


def test_env_reset_step_contract() -> None:
    env = DriveSimEnv()
    obs = env.reset()
    assert "pose" in obs and "grid" in obs and "lidar" in obs
    assert float(obs["grid"].max()) > 0.9

    obs2, reward, done, info = env.step(Action(throttle=0.2, steering=0.0))
    assert "pose" in obs2
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert "distance_to_goal" in info


def test_map_switching_changes_world() -> None:
    env = DriveSimEnv(EnvConfig(map_name="default"))
    default_goal = env.world.goal
    env.set_map("maze")
    assert env.map_name == "maze"
    assert env.world.goal != default_goal


def test_auto_expand_grows_world_near_edge() -> None:
    env = DriveSimEnv(EnvConfig(auto_expand=True))
    state = env.sim.get_state()
    state.vehicle.x = env.world.width - 20.0
    state.vehicle.y = env.world.height - 20.0
    old_width = env.world.width
    old_height = env.world.height

    env.step(Action(throttle=0.0, steering=0.0))

    assert env.world.width > old_width
    assert env.world.height > old_height


def test_randomize_episode_changes_goal_or_start() -> None:
    env = DriveSimEnv()
    old_start = env.world.start
    old_goal = env.world.goal
    obs = env.randomize_episode(np.random.default_rng(5))
    assert "pose" in obs and "goal" in obs
    assert env.world.start != old_start or env.world.goal != old_goal


def test_env_spawns_configured_dynamic_obstacles() -> None:
    env = DriveSimEnv(EnvConfig(dynamic_obstacle_count=3))
    env.reset()
    assert len(env.world.dynamic_obstacles) == 3


def test_reset_seed_reproducible_dynamic_obstacles() -> None:
    env = DriveSimEnv(EnvConfig(dynamic_obstacle_count=2))
    env.reset(seed=123)
    first = [(o.x, o.y, o.vx, o.vy) for o in env.world.dynamic_obstacles]
    env.reset(seed=123)
    second = [(o.x, o.y, o.vx, o.vy) for o in env.world.dynamic_obstacles]
    assert first == second


def test_reset_seed_reproducible_randomized_episode() -> None:
    env = DriveSimEnv(EnvConfig(dynamic_obstacle_count=0))
    env.reset(seed=77)
    env.randomize_episode()
    first = (env.world.start, env.world.goal)
    env.reset(seed=77)
    env.randomize_episode()
    second = (env.world.start, env.world.goal)
    assert first == second


def test_randomize_episode_respawns_dynamic_obstacles_for_new_start_and_goal() -> None:
    env = DriveSimEnv(EnvConfig(dynamic_obstacle_count=3, seed=17))
    env.reset(seed=17)
    env.randomize_episode(np.random.default_rng(123))

    start = env.world.start
    goal = env.world.goal
    assert len(env.world.dynamic_obstacles) == 3
    for obs in env.world.dynamic_obstacles:
        assert np.hypot(obs.x - start[0], obs.y - start[1]) >= 85.0
        assert np.hypot(obs.x - goal[0], obs.y - goal[1]) >= 70.0


def test_randomize_episode_with_explicit_rng_is_fully_reproducible() -> None:
    env_a = DriveSimEnv(EnvConfig(dynamic_obstacle_count=2, seed=17))
    env_b = DriveSimEnv(EnvConfig(dynamic_obstacle_count=2, seed=999))

    env_a.reset()
    env_b.reset()
    env_a.world.dynamic_obstacles = []
    env_b.world.dynamic_obstacles = []
    rng_a = np.random.default_rng(321)
    rng_b = np.random.default_rng(321)
    env_a.randomize_episode(rng=rng_a)
    env_b.randomize_episode(rng=rng_b)

    obstacles_a = [(o.x, o.y, o.w, o.h, o.vx, o.vy) for o in env_a.world.dynamic_obstacles]
    obstacles_b = [(o.x, o.y, o.w, o.h, o.vx, o.vy) for o in env_b.world.dynamic_obstacles]

    assert env_a.world.start == env_b.world.start
    assert env_a.world.goal == env_b.world.goal
    assert obstacles_a == obstacles_b


def test_sensor_driven_mapping_starts_without_baked_obstacles() -> None:
    gt_env = DriveSimEnv(EnvConfig(mapping_mode="ground_truth", dynamic_obstacle_count=0))
    sd_env = DriveSimEnv(EnvConfig(mapping_mode="sensor_driven", dynamic_obstacle_count=0))

    gt_obs = gt_env.reset(seed=9)
    sd_obs = sd_env.reset(seed=9)

    gt_high_conf_occ = int(np.sum(gt_obs["grid"] > 0.95))
    sd_high_conf_occ = int(np.sum(sd_obs["grid"] > 0.95))
    assert gt_high_conf_occ > sd_high_conf_occ


def _require_gym_backend() -> None:
    if importlib.util.find_spec("gymnasium") is None and importlib.util.find_spec("gym") is None:
        pytest.skip("gymnasium/gym is not installed")


def test_gym_wrapper_reset_and_step_contract() -> None:
    _require_gym_backend()
    env = DriveSimGymEnv(EnvConfig(dynamic_obstacle_count=0))

    obs, info = env.reset(seed=4)
    assert isinstance(info, dict)
    assert "pose" in obs and "grid" in obs

    next_obs, reward, terminated, truncated, step_info = env.step(np.array([0.2, 0.0], dtype=np.float32))
    assert "lidar" in next_obs
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "distance_to_goal" in step_info


def test_gym_wrapper_max_steps_sets_truncated() -> None:
    _require_gym_backend()
    env = DriveSimGymEnv(EnvConfig(dynamic_obstacle_count=0, max_steps=1))
    env.reset(seed=3)

    _, _, terminated, truncated, _ = env.step(np.array([0.0, 0.0], dtype=np.float32))

    assert truncated
    assert not terminated


def test_gym_wrapper_clips_actions() -> None:
    _require_gym_backend()
    env = DriveSimGymEnv(EnvConfig(dynamic_obstacle_count=0))
    env.reset(seed=5)

    _, _, _, _, info = env.step(np.array([9.0, -9.0], dtype=np.float32))

    assert info["steps"] == 1


def test_gym_wrapper_can_flatten_observation() -> None:
    _require_gym_backend()
    env = DriveSimGymEnv(EnvConfig(dynamic_obstacle_count=0), observation_mode="flat")

    obs, _ = env.reset(seed=6)

    assert isinstance(obs, np.ndarray)
    assert obs.shape == env.observation_space.shape


def test_gym_wrapper_keeps_fixed_observation_shape_across_maps() -> None:
    _require_gym_backend()
    env = DriveSimGymEnv(
        EnvConfig(dynamic_obstacle_count=0),
        observation_mode="flat",
        map_names=["default", "generated_medium"],
    )

    obs_default, _ = env.reset(seed=7, options={"map_name": "default"})
    obs_generated, _ = env.reset(seed=7, options={"map_name": "generated_medium"})

    assert isinstance(obs_default, np.ndarray)
    assert isinstance(obs_generated, np.ndarray)
    assert obs_default.shape == obs_generated.shape == env.observation_space.shape


def test_gym_wrapper_applies_curriculum_stage_metadata() -> None:
    _require_gym_backend()
    env = DriveSimGymEnv(
        EnvConfig(dynamic_obstacle_count=0),
        observation_mode="flat",
        map_names=["default", "maze"],
    )
    env.apply_curriculum_stage(
        {
            "name": "stage-a",
            "maps": ["maze"],
            "dynamic_obstacle_count": 1,
            "mapping_mode": "sensor_driven",
            "min_goal_distance": 180.0,
        }
    )

    _, info = env.reset(seed=8)

    assert info["map_name"] == "maze"
    assert info["mapping_mode"] == "sensor_driven"
    assert info["curriculum_stage"] == "stage-a"

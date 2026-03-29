from drivesim.core.types import Action
from drivesim.ml.env import DriveSimEnv, EnvConfig
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

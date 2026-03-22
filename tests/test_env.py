from drivesim.core.types import Action
from drivesim.ml.env import DriveSimEnv, EnvConfig


def test_env_reset_step_contract() -> None:
    env = DriveSimEnv()
    obs = env.reset()
    assert "pose" in obs and "grid" in obs and "lidar" in obs

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

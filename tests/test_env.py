from drivesim.core.types import Action
from drivesim.ml.env import DriveSimEnv


def test_env_reset_step_contract() -> None:
    env = DriveSimEnv()
    obs = env.reset()
    assert "pose" in obs and "grid" in obs and "lidar" in obs

    obs2, reward, done, info = env.step(Action(throttle=0.2, steering=0.0))
    assert "pose" in obs2
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert "distance_to_goal" in info

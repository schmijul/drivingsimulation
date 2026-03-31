import numpy as np

from drivesim.ml.live_train import LivePolicyTrainer
from drivesim.ml.policy import LinearPolicy, features_from_observation


def test_live_trainer_progress_updates_after_episode() -> None:
    trainer = LivePolicyTrainer(feature_dim=14)
    obs = {
        "pose": np.zeros(4, dtype=np.float32),
        "goal": np.zeros(2, dtype=np.float32),
        "lidar": np.zeros(11, dtype=np.float32),
        "collided": False,
    }

    trainer.act(obs)
    changed = trainer.observe(
        reward=1.0,
        done=True,
        observation={"collided": False},
        info={"distance_to_goal": 10.0},
    )

    assert changed
    assert trainer.best_score > -1e18


def test_seed_from_linear_policy_preserves_normalization_behavior() -> None:
    obs = {
        "pose": np.array([10.0, 15.0, 0.3, 2.5], dtype=np.float32),
        "goal": np.array([40.0, 25.0], dtype=np.float32),
        "lidar": np.linspace(5.0, 55.0, num=11, dtype=np.float32),
        "collided": False,
    }
    feature_dim = len(features_from_observation(obs))
    policy = LinearPolicy(
        weights=np.full((feature_dim, 2), 0.15, dtype=np.float32),
        bias=np.array([0.05, -0.02], dtype=np.float32),
        feature_mean=np.linspace(-1.0, 1.0, num=feature_dim, dtype=np.float32),
        feature_std=np.linspace(0.5, 1.8, num=feature_dim, dtype=np.float32),
    )
    trainer = LivePolicyTrainer(feature_dim=feature_dim)

    seeded = trainer.seed_from_linear_policy(policy)

    assert seeded
    expected = policy.act(obs)
    actual = trainer.act(obs)
    assert actual.throttle == expected.throttle
    assert actual.steering == expected.steering

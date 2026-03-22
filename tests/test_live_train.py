import numpy as np

from drivesim.ml.live_train import LivePolicyTrainer


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

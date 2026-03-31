from pathlib import Path

import numpy as np

from drivesim.ml.agent import AssistAgent
from drivesim.ml.models import TinyMLPPolicyModel


def test_assist_agent_loads_trained_tiny_mlp_and_changes_action(tmp_path: Path) -> None:
    model_path = tmp_path / "assist_policy.npz"
    model = TinyMLPPolicyModel(
        w1=np.zeros((14, 8), dtype=np.float32),
        b1=np.zeros(8, dtype=np.float32),
        w2=np.zeros((8, 2), dtype=np.float32),
        b2=np.array([-6.0, 6.0], dtype=np.float32),
        feature_mean=np.zeros(14, dtype=np.float32),
        feature_std=np.ones(14, dtype=np.float32),
    )
    model.save(str(model_path))

    trained = AssistAgent(model_path=str(model_path))
    heuristic = AssistAgent(model_path=str(Path(tmp_path) / "missing_model.npz"))

    assert trained.policy is not None
    assert trained.model_name == "tiny_mlp"
    assert trained.mode_label == "assistant (trained:tiny_mlp)"
    assert heuristic.policy is None
    assert heuristic.mode_label == "assistant (heuristic)"

    obs = {
        "pose": np.array([20.0, 20.0, 0.0, 1.0], dtype=np.float32),
        "goal": np.array([200.0, 20.0], dtype=np.float32),
        "lidar": np.full(31, 80.0, dtype=np.float32),
        "lidar_front": [80.0, 80.0, 80.0, 80.0, 80.0],
    }
    trained_action = trained.act(obs)
    heuristic_action = heuristic.act(obs)

    assert -1.0 <= trained_action.throttle <= 1.0
    assert -1.0 <= trained_action.steering <= 1.0
    assert (trained_action.throttle, trained_action.steering) != (
        heuristic_action.throttle,
        heuristic_action.steering,
    )

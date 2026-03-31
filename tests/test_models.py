from pathlib import Path

import numpy as np

from drivesim.ml.models import LinearPolicyModel, TinyMLPPolicyModel, load_policy_model
from drivesim.ml.policy import LinearPolicy


def test_load_linear_model_with_metadata(tmp_path: Path) -> None:
    policy = LinearPolicy(
        weights=np.zeros((11, 2), dtype=np.float32),
        bias=np.zeros(2, dtype=np.float32),
        feature_mean=np.zeros(11, dtype=np.float32),
        feature_std=np.ones(11, dtype=np.float32),
    )
    target = tmp_path / "policy.npz"
    policy.save(str(target))

    loaded = load_policy_model(str(target))
    assert isinstance(loaded, LinearPolicyModel)
    assert loaded.model_name == "linear"


def test_tiny_mlp_round_trip_via_generic_loader(tmp_path: Path) -> None:
    model = TinyMLPPolicyModel(
        w1=np.zeros((14, 6), dtype=np.float32),
        b1=np.zeros(6, dtype=np.float32),
        w2=np.zeros((6, 2), dtype=np.float32),
        b2=np.array([-2.0, 3.0], dtype=np.float32),
        feature_mean=np.zeros(14, dtype=np.float32),
        feature_std=np.ones(14, dtype=np.float32),
    )
    target = tmp_path / "tiny_mlp_policy.npz"
    model.save(str(target))

    loaded = load_policy_model(str(target))
    assert isinstance(loaded, TinyMLPPolicyModel)
    assert loaded.model_name == "tiny_mlp"

    obs = {
        "pose": np.array([5.0, 2.0, 0.0, 1.2], dtype=np.float32),
        "goal": np.array([20.0, 2.0], dtype=np.float32),
        "lidar": np.linspace(10.0, 80.0, num=31, dtype=np.float32),
    }
    expected = model.act(obs)
    actual = loaded.act(obs)
    assert actual.throttle == expected.throttle
    assert actual.steering == expected.steering

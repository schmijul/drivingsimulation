from pathlib import Path

import numpy as np

from drivesim.ml.models import LinearPolicyModel, load_policy_model
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

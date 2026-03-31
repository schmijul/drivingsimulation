import json
from pathlib import Path

import numpy as np

from drivesim.ml.models import LinearPolicyModel, load_policy_model
from drivesim.ml.train import _load_replay, train_model


def test_load_replay_parses_features_and_actions(tmp_path: Path) -> None:
    replay = tmp_path / "mini_replay.jsonl"
    rows = [
        {"features": [1.0, 2.0, 3.0], "action": {"throttle": 0.4, "steering": -0.2}},
        {"features": [3.0, 2.0, 1.0], "action": {"throttle": -0.1, "steering": 0.5}},
    ]
    replay.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    x, y = _load_replay(str(replay))
    assert x.shape == (2, 3)
    assert y.shape == (2, 2)
    assert np.allclose(x[0], np.array([1.0, 2.0, 3.0], dtype=np.float32))
    assert np.allclose(y[1], np.array([-0.1, 0.5], dtype=np.float32))


def test_train_model_writes_linear_policy_with_bias(tmp_path: Path) -> None:
    rng = np.random.default_rng(13)
    true_w = rng.normal(size=(5, 2)).astype(np.float32)
    true_b = np.array([0.35, -0.55], dtype=np.float32)
    x = rng.normal(size=(180, 5)).astype(np.float32)
    y = x @ true_w + true_b

    replay = tmp_path / "replay.jsonl"
    with replay.open("w", encoding="utf-8") as f:
        for idx in range(x.shape[0]):
            row = {
                "features": [float(v) for v in x[idx]],
                "action": {"throttle": float(y[idx, 0]), "steering": float(y[idx, 1])},
            }
            f.write(json.dumps(row) + "\n")

    model_path = tmp_path / "assist_policy.npz"
    train_model(str(replay), str(model_path), l2_reg=1e-6)
    assert model_path.exists()

    loaded = load_policy_model(str(model_path))
    assert isinstance(loaded, LinearPolicyModel)
    pred = loaded.policy.predict(x)
    assert np.allclose(pred, y, atol=1e-3)

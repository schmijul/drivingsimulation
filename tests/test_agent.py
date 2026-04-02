import importlib
from pathlib import Path

import numpy as np

from drivesim.ml.agent import AssistAgent
from drivesim.ml.models import TinyMLPPolicyModel
from drivesim.ml.train_rl import RLTrainConfig, train_rl


def _require_rl_backend() -> None:
    if importlib.util.find_spec("gymnasium") is None:
        import pytest

        pytest.skip("gymnasium is not installed")
    if importlib.util.find_spec("stable_baselines3") is None:
        import pytest

        pytest.skip("stable-baselines3 is not installed")


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


def test_assist_agent_can_load_ppo_and_falls_back_on_unsupported_map(tmp_path: Path) -> None:
    _require_rl_backend()
    result = train_rl(
        RLTrainConfig(
            maps=["default"],
            total_timesteps=64,
            num_envs=1,
            max_steps=80,
            seed=6,
            output_dir=str(tmp_path / "models"),
            curriculum="",
            dynamic_obstacle_count=0,
            ppo_n_steps=32,
            ppo_batch_size=32,
            policy_hidden_sizes=(32, 32),
            eval_episodes_per_map=1,
            track_run=False,
        )
    )

    trained = AssistAgent(model_path=str(result["model_path"]))
    heuristic = AssistAgent(model_path=str(Path(tmp_path) / "missing_model.npz"))

    assert trained.policy is not None
    assert trained.model_name == "ppo"

    supported_obs = {
        "pose": np.array([20.0, 20.0, 0.0, 1.0], dtype=np.float32),
        "goal": np.array([200.0, 20.0], dtype=np.float32),
        "grid": np.zeros((57, 101), dtype=np.float32),
        "lidar": np.full(41, 80.0, dtype=np.float32),
        "lidar_front": [80.0, 80.0, 80.0, 80.0, 80.0],
        "collided": False,
        "map_name": "default",
    }
    unsupported_obs = dict(supported_obs)
    unsupported_obs["map_name"] = "maze"

    trained_supported = trained.act(supported_obs)
    trained_fallback = trained.act(unsupported_obs)
    heuristic_action = heuristic.act(unsupported_obs)

    assert -1.0 <= trained_supported.throttle <= 1.0
    assert -1.0 <= trained_supported.steering <= 1.0
    assert (trained_fallback.throttle, trained_fallback.steering) == (
        heuristic_action.throttle,
        heuristic_action.steering,
    )

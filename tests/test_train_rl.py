import importlib
from pathlib import Path

import pytest

from drivesim.ml.experiment import load_experiment_history
from drivesim.ml.train_rl import RLTrainConfig, default_rl_run_name, train_rl


def _require_rl_backend() -> None:
    if importlib.util.find_spec("gymnasium") is None:
        pytest.skip("gymnasium is not installed")
    if importlib.util.find_spec("stable_baselines3") is None:
        pytest.skip("stable-baselines3 is not installed")


def test_default_rl_run_name_contains_algo_and_seed() -> None:
    name = default_rl_run_name(RLTrainConfig(maps=["default"], seed=17, curriculum="standard"))
    assert name.startswith("ppo_standard_seed17_")


def test_train_rl_smoke_end_to_end(tmp_path) -> None:
    _require_rl_backend()
    history_path = tmp_path / "experiments.jsonl"
    result = train_rl(
        RLTrainConfig(
            maps=["default"],
            total_timesteps=64,
            num_envs=1,
            max_steps=80,
            seed=9,
            output_dir=str(tmp_path / "models"),
            curriculum="",
            dynamic_obstacle_count=0,
            ppo_n_steps=32,
            ppo_batch_size=32,
            policy_hidden_sizes=(32, 32),
            eval_episodes_per_map=1,
            track_run=True,
            experiment_history_path=str(history_path),
        )
    )

    assert Path(result["model_path"]).exists()
    assert Path(result["run_dir"]).exists()
    assert (Path(result["run_dir"]) / "eval.json").exists()
    rows = load_experiment_history(str(history_path))
    assert len(rows) == 1
    assert rows[0]["kind"] == "train_rl"

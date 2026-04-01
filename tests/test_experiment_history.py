from datetime import datetime

from drivesim.ml.experiment import (
    append_experiment_record,
    build_experiment_record,
    format_experiment_row,
    load_experiment_history,
    sort_experiment_history,
)


def test_experiment_history_round_trip_and_sorting(tmp_path) -> None:
    history = tmp_path / "experiments.jsonl"
    append_experiment_record(
        str(history),
        build_experiment_record(
            kind="train_rl",
            name="run-a",
            algo="ppo",
            policy_mode="rl",
            model_path="models/rl/run-a/model.zip",
            seed=11,
            maps=["default"],
            curriculum="standard",
            summary={"episodes": 2.0, "success_rate": 0.4, "avg_total_reward": 1.0},
            config={"timesteps": 128},
            ts=datetime(2026, 4, 1, 19, 15, 0),
        ),
    )
    append_experiment_record(
        str(history),
        build_experiment_record(
            kind="train_rl",
            name="run-b",
            algo="ppo",
            policy_mode="rl",
            model_path="models/rl/run-b/model.zip",
            seed=12,
            maps=["default"],
            curriculum="robust",
            summary={"episodes": 2.0, "success_rate": 0.7, "avg_total_reward": 3.0},
            config={"timesteps": 128},
            ts=datetime(2026, 4, 1, 19, 16, 0),
        ),
    )

    rows = load_experiment_history(str(history))
    by_success = sort_experiment_history(rows, "success")
    by_reward = sort_experiment_history(rows, "reward")

    assert len(rows) == 2
    assert by_success[0]["name"] == "run-b"
    assert by_reward[0]["name"] == "run-b"
    assert "run-b" in format_experiment_row(by_success[0])

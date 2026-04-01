from datetime import datetime
import json

import numpy as np

from drivesim.ml.eval import (
    EvalConfig,
    _append_eval_history,
    default_eval_report_path,
    run_eval,
    run_eval_compare,
)
from drivesim.ml.experiment import load_experiment_history
from drivesim.ml.models import TinyMLPPolicyModel


def test_eval_returns_expected_metrics() -> None:
    summary = run_eval(
        EvalConfig(
            maps=["default"],
            episodes_per_map=2,
            max_steps=120,
            seed=19,
            policy_mode="autopilot",
            dynamic_obstacle_count=1,
        )
    )
    assert summary["episodes"] == 2.0
    assert 0.0 <= summary["success_rate"] <= 1.0
    assert 0.0 <= summary["collision_rate"] <= 1.0
    assert summary["avg_steps"] > 0.0
    assert "default" in summary["per_map"]


def test_eval_is_reproducible_with_same_seed() -> None:
    cfg = EvalConfig(
        maps=["default", "maze"],
        episodes_per_map=1,
        max_steps=140,
        seed=29,
        policy_mode="assistant",
        dynamic_obstacle_count=0,
    )
    first = run_eval(cfg)
    second = run_eval(cfg)
    assert first == second


def test_eval_can_write_json_report(tmp_path) -> None:
    out = tmp_path / "eval_report.json"
    summary = run_eval(
        EvalConfig(
            maps=["default"],
            episodes_per_map=1,
            max_steps=80,
            seed=5,
            policy_mode="autopilot",
            dynamic_obstacle_count=0,
            json_out=str(out),
        )
    )
    assert out.exists()
    assert "default" in summary["per_map"]


def test_eval_assistant_with_trained_model_path_writes_config_and_summary(tmp_path) -> None:
    model_path = tmp_path / "assist_policy.npz"
    model = TinyMLPPolicyModel(
        w1=np.zeros((14, 6), dtype=np.float32),
        b1=np.zeros(6, dtype=np.float32),
        w2=np.zeros((6, 2), dtype=np.float32),
        b2=np.array([0.0, 0.0], dtype=np.float32),
        feature_mean=np.zeros(14, dtype=np.float32),
        feature_std=np.ones(14, dtype=np.float32),
    )
    model.save(str(model_path))

    out = tmp_path / "trained_eval.json"
    summary = run_eval(
        EvalConfig(
            maps=["default"],
            episodes_per_map=1,
            max_steps=80,
            seed=8,
            policy_mode="assistant",
            model_path=str(model_path),
            dynamic_obstacle_count=0,
            mapping_mode="sensor_driven",
            json_out=str(out),
        )
    )
    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["config"]["model_path"] == str(model_path)
    assert payload["config"]["policy_mode"] == "assistant"
    assert payload["config"]["mapping_mode"] == "sensor_driven"
    assert "summary" in payload
    assert "default" in summary["per_map"]


def test_eval_compare_returns_both_and_delta() -> None:
    summary = run_eval_compare(
        EvalConfig(
            maps=["default"],
            episodes_per_map=1,
            max_steps=90,
            seed=17,
            policy_mode="both",
            dynamic_obstacle_count=0,
        )
    )
    assert "assistant" in summary
    assert "autopilot" in summary
    assert "delta_assistant_minus_autopilot" in summary


def test_eval_compare_can_write_json(tmp_path) -> None:
    out = tmp_path / "eval_compare.json"
    run_eval_compare(
        EvalConfig(
            maps=["default"],
            episodes_per_map=1,
            max_steps=80,
            seed=23,
            policy_mode="both",
            dynamic_obstacle_count=0,
            json_out=str(out),
        )
    )
    assert out.exists()


def test_default_eval_report_path_contains_policy_seed_and_timestamp() -> None:
    path = default_eval_report_path("both", 11, now=datetime(2026, 3, 26, 14, 5, 7))
    assert path == "replays/evals/eval_both_seed11_20260326-140507.json"


def test_append_eval_history_writes_jsonl(tmp_path) -> None:
    history = tmp_path / "index.jsonl"
    cfg = EvalConfig(maps=["default"], policy_mode="assistant", seed=11)
    _append_eval_history(history, "replays/evals/eval_assistant_seed11_x.json", cfg, {"success_rate": 0.5})
    lines = history.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["policy_mode"] == "assistant"
    assert row["summary"]["success_rate"] == 0.5


def test_eval_curriculum_adds_stage_summaries() -> None:
    summary = run_eval(
        EvalConfig(
            maps=["default", "blocks"],
            episodes_per_map=1,
            max_steps=80,
            seed=13,
            policy_mode="autopilot",
            dynamic_obstacle_count=0,
            curriculum="easy",
        )
    )
    assert summary["curriculum"] == "easy"
    assert "per_stage" in summary
    assert "easy-start" in summary["per_stage"]


def test_eval_can_track_experiment_history(tmp_path) -> None:
    history_path = tmp_path / "experiments.jsonl"
    summary = run_eval(
        EvalConfig(
            maps=["default"],
            episodes_per_map=1,
            max_steps=80,
            seed=31,
            policy_mode="autopilot",
            dynamic_obstacle_count=0,
            track_run=True,
            experiment_history_path=str(history_path),
        )
    )
    rows = load_experiment_history(str(history_path))
    assert len(rows) == 1
    assert rows[0]["kind"] == "eval"
    assert rows[0]["policy_mode"] == "autopilot"
    assert rows[0]["summary"]["success_rate"] == summary["success_rate"]

from datetime import datetime

from drivesim.ml.eval import EvalConfig, default_eval_report_path, run_eval, run_eval_compare


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

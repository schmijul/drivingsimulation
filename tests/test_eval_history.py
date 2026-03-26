import json

from drivesim.ml.eval_history import format_row, load_history, sort_history


def test_load_and_sort_history(tmp_path) -> None:
    history = tmp_path / "index.jsonl"
    rows = [
        {
            "ts": "2026-03-26T10:00:00",
            "policy_mode": "assistant",
            "report_path": "a.json",
            "summary": {"success_rate": 0.2, "collision_rate": 0.5, "avg_total_reward": 1.0, "episodes": 3.0},
        },
        {
            "ts": "2026-03-26T10:10:00",
            "policy_mode": "both",
            "report_path": "b.json",
            "summary": {"success_rate": 0.8, "collision_rate": 0.1, "avg_total_reward": 5.0, "episodes": 3.0},
        },
    ]
    history.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    loaded = load_history(str(history))
    assert len(loaded) == 2
    by_latest = sort_history(loaded, "latest")
    assert by_latest[0]["report_path"] == "b.json"
    by_success = sort_history(loaded, "success")
    assert by_success[0]["report_path"] == "b.json"


def test_format_row_contains_key_fields() -> None:
    row = {
        "ts": "2026-03-26T11:00:00",
        "policy_mode": "autopilot",
        "report_path": "x.json",
        "summary": {"success_rate": 0.5, "collision_rate": 0.2, "avg_total_reward": 2.25, "episodes": 4.0},
    }
    line = format_row(row)
    assert "autopilot" in line
    assert "succ=" in line
    assert "x.json" in line

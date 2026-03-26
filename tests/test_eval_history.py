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
            "summary": {
                "assistant": {"success_rate": 0.8, "collision_rate": 0.1, "avg_total_reward": 5.0, "episodes": 3.0},
                "autopilot": {"success_rate": 0.6, "collision_rate": 0.2, "avg_total_reward": 3.5, "episodes": 3.0},
                "delta_assistant_minus_autopilot": {
                    "success_rate": 0.2,
                    "collision_rate": -0.1,
                    "avg_total_reward": 1.5,
                },
            },
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


def test_format_row_includes_delta_for_compare_rows() -> None:
    row = {
        "ts": "2026-03-26T12:00:00",
        "policy_mode": "both",
        "report_path": "cmp.json",
        "summary": {
            "assistant": {"success_rate": 0.5, "collision_rate": 0.25, "avg_total_reward": 2.0, "episodes": 4.0},
            "autopilot": {"success_rate": 0.3, "collision_rate": 0.4, "avg_total_reward": 1.0, "episodes": 4.0},
            "delta_assistant_minus_autopilot": {
                "success_rate": 0.2,
                "collision_rate": -0.15,
                "avg_total_reward": 1.0,
            },
        },
    }
    line = format_row(row)
    assert "ds=" in line
    assert "dc=" in line
    assert "dr=" in line


def test_best_row_is_highest_success() -> None:
    rows = [
        {"summary": {"success_rate": 0.25}, "report_path": "a.json"},
        {"summary": {"success_rate": 0.9}, "report_path": "b.json"},
        {"summary": {"success_rate": 0.6}, "report_path": "c.json"},
    ]
    best = sort_history(rows, "success")[0]
    assert best["report_path"] == "b.json"


def test_success_and_map_filter_logic() -> None:
    rows = [
        {"maps": ["default"], "summary": {"success_rate": 0.2}, "report_path": "a.json"},
        {"maps": ["maze"], "summary": {"success_rate": 0.7}, "report_path": "b.json"},
        {"maps": ["default", "maze"], "summary": {"success_rate": 0.9}, "report_path": "c.json"},
    ]
    filtered = [r for r in rows if float(r["summary"]["success_rate"]) >= 0.7]
    filtered = [r for r in filtered if "maze" in r["maps"]]
    assert [r["report_path"] for r in filtered] == ["b.json", "c.json"]

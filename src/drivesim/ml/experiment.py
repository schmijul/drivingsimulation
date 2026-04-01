from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any


def default_experiment_history_path() -> str:
    return "replays/experiments/index.jsonl"


def append_experiment_record(path: str, row: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def build_experiment_record(
    *,
    kind: str,
    name: str,
    algo: str,
    policy_mode: str,
    model_path: str,
    seed: int,
    maps: list[str],
    curriculum: str,
    summary: dict[str, Any],
    config: dict[str, Any],
    report_path: str = "",
    ts: datetime | None = None,
) -> dict[str, Any]:
    return {
        "ts": (ts or datetime.now()).isoformat(timespec="seconds"),
        "kind": kind,
        "name": name,
        "algo": algo,
        "policy_mode": policy_mode,
        "model_path": model_path,
        "seed": seed,
        "maps": maps,
        "curriculum": curriculum,
        "report_path": report_path,
        "summary": summary,
        "config": config,
    }


def load_experiment_history(path: str) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []

    rows: list[dict[str, Any]] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def summary_block(row: dict[str, Any]) -> dict[str, Any]:
    summary = row.get("summary", {})
    if isinstance(summary, dict):
        return summary
    return {}


def sort_experiment_history(rows: list[dict[str, Any]], sort_by: str) -> list[dict[str, Any]]:
    if sort_by == "success":
        return sorted(rows, key=lambda row: float(summary_block(row).get("success_rate", 0.0)), reverse=True)
    if sort_by == "reward":
        return sorted(rows, key=lambda row: float(summary_block(row).get("avg_total_reward", 0.0)), reverse=True)
    return sorted(rows, key=lambda row: str(row.get("ts", "")), reverse=True)


def format_experiment_row(row: dict[str, Any]) -> str:
    summary = summary_block(row)
    success = float(summary.get("success_rate", 0.0))
    reward = float(summary.get("avg_total_reward", 0.0))
    episodes = int(float(summary.get("episodes", 0.0)))
    return (
        f"{str(row.get('ts', '-')):<19}  "
        f"{str(row.get('kind', '-')):<9}  "
        f"{str(row.get('algo', '-')):<5}  "
        f"{str(row.get('curriculum', '-')):<10}  "
        f"ep={episodes:<3d}  "
        f"succ={success:5.1%}  "
        f"rew={reward:8.3f}  "
        f"{str(row.get('name', '-'))}"
    )

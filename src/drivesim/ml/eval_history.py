from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_history(path: str) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def sort_history(rows: list[dict[str, Any]], sort_by: str) -> list[dict[str, Any]]:
    if sort_by == "success":
        return sorted(
            rows,
            key=lambda r: float(r.get("summary", {}).get("success_rate", 0.0)),
            reverse=True,
        )
    return sorted(rows, key=lambda r: str(r.get("ts", "")), reverse=True)


def format_row(row: dict[str, Any]) -> str:
    summary = row.get("summary", {})
    success = float(summary.get("success_rate", 0.0))
    collision = float(summary.get("collision_rate", 0.0))
    reward = float(summary.get("avg_total_reward", 0.0))
    episodes = int(float(summary.get("episodes", 0.0)))
    return (
        f"{row.get('ts', '-'):<19}  "
        f"{str(row.get('policy_mode', '-')):<10}  "
        f"ep={episodes:<3d}  "
        f"succ={success:5.1%}  "
        f"coll={collision:5.1%}  "
        f"rew={reward:8.3f}  "
        f"{row.get('report_path', '-')}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Show eval history from replays/evals/index.jsonl")
    parser.add_argument("--path", default="replays/evals/index.jsonl", help="History jsonl path")
    parser.add_argument("--limit", type=int, default=10, help="Rows to show")
    parser.add_argument("--sort", choices=["latest", "success"], default="latest", help="Sort order")
    parser.add_argument("--policy", default="", help="Optional filter (assistant/autopilot/both)")
    parser.add_argument("--best", action="store_true", help="Show only the single best row by success rate")
    args = parser.parse_args()

    rows = load_history(args.path)
    if args.policy:
        rows = [r for r in rows if str(r.get("policy_mode", "")) == args.policy]
    rows = sort_history(rows, args.sort)
    if args.best:
        rows = sort_history(rows, "success")[:1]
    else:
        rows = rows[: max(0, args.limit)]

    if not rows:
        print("no eval history entries")
        return

    print("timestamp             policy      episodes  success   collision   reward     report")
    for row in rows:
        print(format_row(row))


if __name__ == "__main__":
    main()

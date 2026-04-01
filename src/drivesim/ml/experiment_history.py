from __future__ import annotations

import argparse

from drivesim.ml.experiment import (
    default_experiment_history_path,
    format_experiment_row,
    load_experiment_history,
    sort_experiment_history,
    summary_block,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Show tracked training and evaluation runs.")
    parser.add_argument("--path", default=default_experiment_history_path(), help="History jsonl path")
    parser.add_argument("--limit", type=int, default=10, help="Rows to show")
    parser.add_argument("--sort", choices=["latest", "success", "reward"], default="latest", help="Sort order")
    parser.add_argument("--kind", default="", help="Optional kind filter, for example train_rl or eval")
    parser.add_argument("--algo", default="", help="Optional algorithm filter, for example ppo")
    parser.add_argument("--curriculum", default="", help="Optional curriculum filter")
    parser.add_argument("--policy", default="", help="Optional policy filter, for example rl or assistant")
    parser.add_argument("--map", default="", help="Optional map filter")
    parser.add_argument("--min-success", type=float, default=-1.0, help="Filter by minimum success rate in [0,1]")
    parser.add_argument("--best", action="store_true", help="Show only the single best row by success rate")
    args = parser.parse_args()

    rows = load_experiment_history(args.path)
    if args.kind:
        rows = [row for row in rows if str(row.get("kind", "")) == args.kind]
    if args.algo:
        rows = [row for row in rows if str(row.get("algo", "")) == args.algo]
    if args.curriculum:
        rows = [row for row in rows if str(row.get("curriculum", "")) == args.curriculum]
    if args.policy:
        rows = [row for row in rows if str(row.get("policy_mode", "")) == args.policy]
    if args.map:
        rows = [row for row in rows if args.map in list(row.get("maps", []))]
    if args.min_success >= 0.0:
        rows = [row for row in rows if float(summary_block(row).get("success_rate", 0.0)) >= args.min_success]

    rows = sort_experiment_history(rows, "success" if args.best else args.sort)
    rows = rows[:1] if args.best else rows[: max(0, args.limit)]
    if not rows:
        print("no experiment history entries")
        return

    print("timestamp             kind       algo   curriculum  episodes  success   reward     name")
    for row in rows:
        print(format_experiment_row(row))


if __name__ == "__main__":
    main()

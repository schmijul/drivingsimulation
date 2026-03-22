from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from drivesim.ml.policy import fit_linear_policy


def _load_replay(path: str) -> tuple[np.ndarray, np.ndarray]:
    feature_rows: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []

    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            feature_rows.append(np.asarray(row["features"], dtype=np.float32))
            action = row["action"]
            action_rows.append(np.asarray([action["throttle"], action["steering"]], dtype=np.float32))

    if not feature_rows:
        raise ValueError("Replay file contains no training rows with features.")

    return np.vstack(feature_rows), np.vstack(action_rows)


def train_model(replay_path: str, output_path: str, l2_reg: float) -> None:
    features, actions = _load_replay(replay_path)
    policy = fit_linear_policy(features, actions, l2_reg=l2_reg)
    policy.save(output_path)
    print(
        f"trained policy from {features.shape[0]} samples, "
        f"{features.shape[1]} features -> {output_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a simple driving assistant policy from replay data.")
    parser.add_argument("--replay", default="replays/latest_episode.jsonl", help="Path to replay JSONL file")
    parser.add_argument("--output", default="models/assist_policy.npz", help="Output path for trained policy")
    parser.add_argument("--l2", type=float, default=1e-2, help="L2 regularization strength")
    args = parser.parse_args()
    train_model(args.replay, args.output, args.l2)


if __name__ == "__main__":
    main()

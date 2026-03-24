from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from drivesim.ml.agent import AssistAgent
from drivesim.ml.env import DriveSimEnv, EnvConfig


@dataclass
class EvalConfig:
    maps: list[str]
    episodes_per_map: int = 8
    max_steps: int = 800
    seed: int = 11
    policy_mode: str = "assistant"
    model_path: str = "models/assist_policy.npz"
    dynamic_obstacle_count: int = 2


def _episode_metrics(env: DriveSimEnv, policy_mode: str, agent: AssistAgent, episode_seed: int) -> dict[str, float | bool | int]:
    env.reset(seed=episode_seed)
    obs = env.randomize_episode()
    done = False
    total_reward = 0.0
    steps = 0
    info = {"distance_to_goal": float("inf")}
    collided = False

    while not done:
        if policy_mode == "autopilot":
            action = env.autopilot_action(env.sim.get_state())
        else:
            action = agent.act(obs)
        obs, reward, done, info = env.step(action)
        total_reward += float(reward)
        steps += 1
        collided = bool(obs["collided"])

    distance = float(info["distance_to_goal"])
    success = distance < 18.0 and not collided
    return {
        "success": success,
        "collided": collided,
        "distance": distance,
        "steps": steps,
        "total_reward": total_reward,
    }


def run_eval(cfg: EvalConfig) -> dict[str, float]:
    rng = np.random.default_rng(cfg.seed)
    agent = AssistAgent(cfg.model_path)
    all_rows: list[dict[str, float | bool | int]] = []

    for map_name in cfg.maps:
        env = DriveSimEnv(
            EnvConfig(
                map_name=map_name,
                max_steps=cfg.max_steps,
                auto_expand=False,
                dynamic_obstacle_count=cfg.dynamic_obstacle_count,
                seed=cfg.seed,
            )
        )
        for _ in range(cfg.episodes_per_map):
            episode_seed = int(rng.integers(0, 2**31 - 1))
            row = _episode_metrics(env, cfg.policy_mode, agent, episode_seed)
            all_rows.append(row)

    total = max(1, len(all_rows))
    success_rate = sum(1 for r in all_rows if bool(r["success"])) / total
    collision_rate = sum(1 for r in all_rows if bool(r["collided"])) / total
    avg_distance = float(np.mean([float(r["distance"]) for r in all_rows])) if all_rows else float("nan")
    avg_steps = float(np.mean([float(r["steps"]) for r in all_rows])) if all_rows else float("nan")
    avg_reward = float(np.mean([float(r["total_reward"]) for r in all_rows])) if all_rows else float("nan")

    return {
        "episodes": float(total),
        "success_rate": success_rate,
        "collision_rate": collision_rate,
        "avg_distance_to_goal": avg_distance,
        "avg_steps": avg_steps,
        "avg_total_reward": avg_reward,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Headless policy evaluation across fixed maps and seeds.")
    parser.add_argument("--maps", default="default,maze,blocks", help="Comma-separated map list")
    parser.add_argument("--episodes", type=int, default=8, help="Episodes per map")
    parser.add_argument("--max-steps", type=int, default=800, help="Max steps per episode")
    parser.add_argument("--seed", type=int, default=11, help="Evaluation seed")
    parser.add_argument(
        "--policy",
        choices=["assistant", "autopilot"],
        default="assistant",
        help="Policy to evaluate",
    )
    parser.add_argument("--model", default="models/assist_policy.npz", help="Assistant model path")
    parser.add_argument("--dynamic-obstacles", type=int, default=2, help="Dynamic obstacles per episode")
    args = parser.parse_args()

    maps = [m.strip() for m in args.maps.split(",") if m.strip()]
    cfg = EvalConfig(
        maps=maps,
        episodes_per_map=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        policy_mode=args.policy,
        model_path=args.model,
        dynamic_obstacle_count=args.dynamic_obstacles,
    )
    summary = run_eval(cfg)
    print(
        "eval "
        f"episodes={int(summary['episodes'])} "
        f"success={summary['success_rate']:.1%} "
        f"collision={summary['collision_rate']:.1%} "
        f"avg_dist={summary['avg_distance_to_goal']:.1f} "
        f"avg_steps={summary['avg_steps']:.1f} "
        f"avg_reward={summary['avg_total_reward']:.3f}"
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np

from drivesim.ml.agent import AssistAgent
from drivesim.ml.curriculum import resolve_curriculum, single_stage_curriculum
from drivesim.ml.env import DriveSimEnv, DriveSimGymEnv, EnvConfig
from drivesim.ml.experiment import (
    append_experiment_record,
    build_experiment_record,
    default_experiment_history_path,
)
from drivesim.ml.rl_utils import load_rl_model


@dataclass
class EvalConfig:
    maps: list[str]
    episodes_per_map: int = 8
    max_steps: int = 800
    seed: int = 11
    policy_mode: str = "assistant"
    model_path: str = "models/assist_policy.npz"
    dynamic_obstacle_count: int = 2
    mapping_mode: str = "ground_truth"
    min_goal_distance: float = 160.0
    curriculum: str = ""
    rl_algorithm: str = "ppo"
    json_out: str = ""
    track_run: bool = False
    experiment_history_path: str = default_experiment_history_path()


def default_eval_report_path(policy_mode: str, seed: int, now: datetime | None = None) -> str:
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    safe_policy = policy_mode.replace("/", "-")
    return f"replays/evals/eval_{safe_policy}_seed{seed}_{stamp}.json"


def _append_eval_history(
    history_path: Path,
    report_path: str,
    cfg: EvalConfig,
    summary: dict[str, object],
    now: datetime | None = None,
) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "ts": (now or datetime.now()).isoformat(timespec="seconds"),
        "report_path": report_path,
        "policy_mode": cfg.policy_mode,
        "seed": cfg.seed,
        "maps": cfg.maps,
        "episodes_per_map": cfg.episodes_per_map,
        "curriculum": cfg.curriculum,
        "summary": summary,
    }
    with history_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def _write_eval_json(path: str, cfg: EvalConfig, summary: dict[str, object]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "config": {
                    "maps": cfg.maps,
                    "episodes_per_map": cfg.episodes_per_map,
                    "max_steps": cfg.max_steps,
                    "seed": cfg.seed,
                    "policy_mode": cfg.policy_mode,
                    "model_path": cfg.model_path,
                    "dynamic_obstacle_count": cfg.dynamic_obstacle_count,
                    "mapping_mode": cfg.mapping_mode,
                    "min_goal_distance": cfg.min_goal_distance,
                    "curriculum": cfg.curriculum,
                    "rl_algorithm": cfg.rl_algorithm,
                },
                "summary": summary,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _maybe_track_eval_run(cfg: EvalConfig, summary: dict[str, object], kind: str) -> None:
    if not cfg.track_run:
        return

    config_payload = {
        "maps": cfg.maps,
        "episodes_per_map": cfg.episodes_per_map,
        "max_steps": cfg.max_steps,
        "seed": cfg.seed,
        "policy_mode": cfg.policy_mode,
        "model_path": cfg.model_path,
        "dynamic_obstacle_count": cfg.dynamic_obstacle_count,
        "mapping_mode": cfg.mapping_mode,
        "min_goal_distance": cfg.min_goal_distance,
        "curriculum": cfg.curriculum,
        "rl_algorithm": cfg.rl_algorithm,
    }
    row = build_experiment_record(
        kind=kind,
        name=f"{cfg.policy_mode}:{cfg.curriculum or 'custom'}",
        algo=cfg.rl_algorithm if cfg.policy_mode == "rl" else cfg.policy_mode,
        policy_mode=cfg.policy_mode,
        model_path=cfg.model_path,
        seed=cfg.seed,
        maps=cfg.maps,
        curriculum=cfg.curriculum or "custom",
        summary=summary,
        config=config_payload,
        report_path=cfg.json_out,
    )
    append_experiment_record(cfg.experiment_history_path, row)


def _episode_metrics(
    env: DriveSimEnv,
    policy_mode: str,
    agent: AssistAgent,
    episode_seed: int,
    min_goal_distance: float,
) -> dict[str, float | bool | int | str]:
    env.reset(seed=episode_seed)
    obs = env.randomize_episode(min_goal_distance=min_goal_distance)
    done = False
    total_reward = 0.0
    steps = 0
    info = {"distance_to_goal": float("inf"), "goal_reached": False, "map_name": env.map_name}
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
    success = bool(info["goal_reached"]) and not collided
    return {
        "success": success,
        "collided": collided,
        "distance": distance,
        "steps": steps,
        "total_reward": total_reward,
        "map_name": str(info["map_name"]),
    }


def _episode_metrics_rl(
    env: DriveSimGymEnv,
    model: Any,
    episode_seed: int,
) -> dict[str, float | bool | int | str]:
    obs, reset_info = env.reset(seed=episode_seed)
    terminated = False
    truncated = False
    total_reward = 0.0
    steps = 0
    info: dict[str, Any] = {"distance_to_goal": float("inf"), "goal_reached": False, "map_name": reset_info["map_name"]}

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps += 1

    distance = float(info["distance_to_goal"])
    collided = bool(info["collided"])
    success = bool(info["goal_reached"]) and not collided
    return {
        "success": success,
        "collided": collided,
        "distance": distance,
        "steps": steps,
        "total_reward": total_reward,
        "map_name": str(info["map_name"]),
    }


def _summarize(rows: list[dict[str, float | bool | int | str]]) -> dict[str, float]:
    total = max(1, len(rows))
    success_rate = sum(1 for row in rows if bool(row["success"])) / total
    collision_rate = sum(1 for row in rows if bool(row["collided"])) / total
    avg_distance = float(np.mean([float(row["distance"]) for row in rows])) if rows else float("nan")
    avg_steps = float(np.mean([float(row["steps"]) for row in rows])) if rows else float("nan")
    avg_reward = float(np.mean([float(row["total_reward"]) for row in rows])) if rows else float("nan")
    return {
        "episodes": float(total),
        "success_rate": success_rate,
        "collision_rate": collision_rate,
        "avg_distance_to_goal": avg_distance,
        "avg_steps": avg_steps,
        "avg_total_reward": avg_reward,
    }


def run_eval(cfg: EvalConfig) -> dict[str, object]:
    rng = np.random.default_rng(cfg.seed)
    curriculum = (
        resolve_curriculum(cfg.curriculum, maps_override=cfg.maps)
        if cfg.curriculum
        else single_stage_curriculum(
            maps=cfg.maps,
            dynamic_obstacle_count=cfg.dynamic_obstacle_count,
            mapping_mode=cfg.mapping_mode,
            min_goal_distance=cfg.min_goal_distance,
        )
    )

    all_rows: list[dict[str, float | bool | int | str]] = []
    rows_by_map: dict[str, list[dict[str, float | bool | int | str]]] = {}
    rows_by_stage: dict[str, list[dict[str, float | bool | int | str]]] = {}

    if cfg.policy_mode == "rl":
        model = load_rl_model(cfg.model_path, cfg.rl_algorithm)
        allowed_maps = sorted({map_name for stage in curriculum.stages for map_name in stage.maps})
        env = DriveSimGymEnv(
            EnvConfig(
                map_name=allowed_maps[0],
                max_steps=cfg.max_steps,
                auto_expand=False,
                dynamic_obstacle_count=cfg.dynamic_obstacle_count,
                seed=cfg.seed,
                mapping_mode=cfg.mapping_mode,
            ),
            observation_mode="flat",
            map_names=allowed_maps,
            randomize_on_reset=True,
            min_goal_distance=cfg.min_goal_distance,
        )
        for stage in curriculum.stages:
            for map_name in stage.maps:
                env.set_episode_profile(
                    map_names=[map_name],
                    dynamic_obstacle_count=stage.dynamic_obstacle_count,
                    mapping_mode=stage.mapping_mode,
                    min_goal_distance=stage.min_goal_distance,
                    randomize_on_reset=True,
                    stage_name=stage.name,
                )
                for _ in range(cfg.episodes_per_map):
                    episode_seed = int(rng.integers(0, 2**31 - 1))
                    row = _episode_metrics_rl(env, model, episode_seed)
                    all_rows.append(row)
                    rows_by_map.setdefault(map_name, []).append(row)
                    rows_by_stage.setdefault(stage.name, []).append(row)
    else:
        agent = AssistAgent(cfg.model_path)
        for stage in curriculum.stages:
            env = DriveSimEnv(
                EnvConfig(
                    map_name=stage.maps[0],
                    max_steps=cfg.max_steps,
                    auto_expand=False,
                    dynamic_obstacle_count=stage.dynamic_obstacle_count,
                    seed=cfg.seed,
                    mapping_mode=stage.mapping_mode,
                )
            )
            env.set_mapping_mode(stage.mapping_mode)
            env.dynamic_obstacle_count = stage.dynamic_obstacle_count
            env.config.dynamic_obstacle_count = stage.dynamic_obstacle_count
            for map_name in stage.maps:
                env.set_map(map_name)
                env.set_mapping_mode(stage.mapping_mode)
                env.dynamic_obstacle_count = stage.dynamic_obstacle_count
                env.config.dynamic_obstacle_count = stage.dynamic_obstacle_count
                for _ in range(cfg.episodes_per_map):
                    episode_seed = int(rng.integers(0, 2**31 - 1))
                    row = _episode_metrics(
                        env,
                        cfg.policy_mode,
                        agent,
                        episode_seed,
                        min_goal_distance=stage.min_goal_distance,
                    )
                    all_rows.append(row)
                    rows_by_map.setdefault(map_name, []).append(row)
                    rows_by_stage.setdefault(stage.name, []).append(row)

    overall = _summarize(all_rows)
    per_map = {map_name: _summarize(rows) for map_name, rows in rows_by_map.items()}
    per_stage = {stage_name: _summarize(rows) for stage_name, rows in rows_by_stage.items()}
    result: dict[str, object] = {
        **overall,
        "curriculum": curriculum.name,
        "per_map": per_map,
        "per_stage": per_stage,
    }

    if cfg.json_out:
        _write_eval_json(cfg.json_out, cfg, result)
        _append_eval_history(Path("replays/evals/index.jsonl"), cfg.json_out, cfg, result)

    _maybe_track_eval_run(cfg, result, kind="eval")
    return result


def run_eval_compare(cfg: EvalConfig) -> dict[str, object]:
    assistant_cfg = EvalConfig(
        maps=cfg.maps,
        episodes_per_map=cfg.episodes_per_map,
        max_steps=cfg.max_steps,
        seed=cfg.seed,
        policy_mode="assistant",
        model_path=cfg.model_path,
        dynamic_obstacle_count=cfg.dynamic_obstacle_count,
        mapping_mode=cfg.mapping_mode,
        min_goal_distance=cfg.min_goal_distance,
        curriculum=cfg.curriculum,
        rl_algorithm=cfg.rl_algorithm,
        json_out="",
        track_run=False,
        experiment_history_path=cfg.experiment_history_path,
    )
    autopilot_cfg = EvalConfig(
        maps=cfg.maps,
        episodes_per_map=cfg.episodes_per_map,
        max_steps=cfg.max_steps,
        seed=cfg.seed,
        policy_mode="autopilot",
        model_path=cfg.model_path,
        dynamic_obstacle_count=cfg.dynamic_obstacle_count,
        mapping_mode=cfg.mapping_mode,
        min_goal_distance=cfg.min_goal_distance,
        curriculum=cfg.curriculum,
        rl_algorithm=cfg.rl_algorithm,
        json_out="",
        track_run=False,
        experiment_history_path=cfg.experiment_history_path,
    )
    assistant = run_eval(assistant_cfg)
    autopilot = run_eval(autopilot_cfg)
    delta = {
        "success_rate": float(assistant["success_rate"]) - float(autopilot["success_rate"]),
        "collision_rate": float(assistant["collision_rate"]) - float(autopilot["collision_rate"]),
        "avg_distance_to_goal": float(assistant["avg_distance_to_goal"]) - float(autopilot["avg_distance_to_goal"]),
        "avg_steps": float(assistant["avg_steps"]) - float(autopilot["avg_steps"]),
        "avg_total_reward": float(assistant["avg_total_reward"]) - float(autopilot["avg_total_reward"]),
    }
    result: dict[str, object] = {
        "curriculum": assistant["curriculum"],
        "assistant": assistant,
        "autopilot": autopilot,
        "delta_assistant_minus_autopilot": delta,
    }
    if cfg.json_out:
        _write_eval_json(cfg.json_out, cfg, result)
        history_cfg = EvalConfig(
            maps=cfg.maps,
            episodes_per_map=cfg.episodes_per_map,
            max_steps=cfg.max_steps,
            seed=cfg.seed,
            policy_mode="both",
            model_path=cfg.model_path,
            dynamic_obstacle_count=cfg.dynamic_obstacle_count,
            mapping_mode=cfg.mapping_mode,
            min_goal_distance=cfg.min_goal_distance,
            curriculum=cfg.curriculum,
            rl_algorithm=cfg.rl_algorithm,
            json_out=cfg.json_out,
            track_run=False,
            experiment_history_path=cfg.experiment_history_path,
        )
        _append_eval_history(Path("replays/evals/index.jsonl"), cfg.json_out, history_cfg, result)

    _maybe_track_eval_run(cfg, result, kind="eval_compare")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Headless policy evaluation across fixed maps and seeds.")
    parser.add_argument("--maps", default="default,maze,blocks", help="Comma-separated map list")
    parser.add_argument("--episodes", type=int, default=8, help="Episodes per map")
    parser.add_argument("--max-steps", type=int, default=800, help="Max steps per episode")
    parser.add_argument("--seed", type=int, default=11, help="Evaluation seed")
    parser.add_argument(
        "--policy",
        choices=["assistant", "autopilot", "rl", "both"],
        default="assistant",
        help="Policy to evaluate",
    )
    parser.add_argument("--model", default="models/assist_policy.npz", help="Policy model path")
    parser.add_argument("--dynamic-obstacles", type=int, default=2, help="Dynamic obstacles per episode")
    parser.add_argument(
        "--mapping-mode",
        choices=["ground_truth", "sensor_driven"],
        default="ground_truth",
        help="Occupancy mapping mode",
    )
    parser.add_argument("--min-goal-distance", type=float, default=160.0, help="Minimum reset goal distance")
    parser.add_argument(
        "--curriculum",
        choices=["", "easy", "standard", "robust"],
        default="",
        help="Optional curriculum preset for staged evaluation",
    )
    parser.add_argument(
        "--rl-algo",
        choices=["ppo"],
        default="ppo",
        help="RL algorithm loader to use when --policy rl",
    )
    parser.add_argument("--json-out", default="", help="Optional path to write evaluation summary JSON")
    parser.add_argument(
        "--json-auto",
        action="store_true",
        help="Auto-write JSON report to replays/evals with a timestamped filename",
    )
    parser.add_argument(
        "--no-track-run",
        action="store_true",
        help="Disable experiment history tracking for this CLI invocation",
    )
    args = parser.parse_args()

    json_out = args.json_out
    if args.json_auto and not json_out:
        json_out = default_eval_report_path(args.policy, args.seed)

    maps = [map_name.strip() for map_name in args.maps.split(",") if map_name.strip()]
    cfg = EvalConfig(
        maps=maps,
        episodes_per_map=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        policy_mode=args.policy,
        model_path=args.model,
        dynamic_obstacle_count=args.dynamic_obstacles,
        mapping_mode=args.mapping_mode,
        min_goal_distance=args.min_goal_distance,
        curriculum=args.curriculum,
        rl_algorithm=args.rl_algo,
        json_out=json_out,
        track_run=not args.no_track_run,
    )
    if cfg.policy_mode == "both":
        summary = run_eval_compare(cfg)
        assistant = summary["assistant"]
        autopilot = summary["autopilot"]
        delta = summary["delta_assistant_minus_autopilot"]
        print(
            "eval assistant "
            f"episodes={int(assistant['episodes'])} "
            f"success={assistant['success_rate']:.1%} "
            f"collision={assistant['collision_rate']:.1%} "
            f"avg_dist={assistant['avg_distance_to_goal']:.1f} "
            f"avg_steps={assistant['avg_steps']:.1f} "
            f"avg_reward={assistant['avg_total_reward']:.3f}"
        )
        print(
            "eval autopilot "
            f"episodes={int(autopilot['episodes'])} "
            f"success={autopilot['success_rate']:.1%} "
            f"collision={autopilot['collision_rate']:.1%} "
            f"avg_dist={autopilot['avg_distance_to_goal']:.1f} "
            f"avg_steps={autopilot['avg_steps']:.1f} "
            f"avg_reward={autopilot['avg_total_reward']:.3f}"
        )
        print(
            "eval delta(assistant-autopilot) "
            f"success={delta['success_rate']:+.1%} "
            f"collision={delta['collision_rate']:+.1%} "
            f"avg_dist={delta['avg_distance_to_goal']:+.1f} "
            f"avg_steps={delta['avg_steps']:+.1f} "
            f"avg_reward={delta['avg_total_reward']:+.3f}"
        )
    else:
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

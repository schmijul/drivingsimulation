from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from drivesim.core.types import Action
from drivesim.ml.env import DriveSimEnv, EnvConfig
from drivesim.ml.eval import EvalConfig, run_eval
from drivesim.ml.models import TinyMLPPolicyModel
from drivesim.ml.policy import LinearPolicy, features_from_observation, fit_linear_policy


@dataclass
class TrainAnyoneConfig:
    maps: list[str]
    seed: int = 21
    output_path: str = "models/assist_policy.npz"
    max_steps: int = 600
    min_goal_distance: float = 180.0
    dynamic_obstacle_count: int = 0
    l2_reg: float = 5e-3
    bc_episodes_per_map: int = 4
    dagger_rounds: int = 1
    dagger_episodes_per_map: int = 2
    teacher_mix_start: float = 0.55
    teacher_mix_end: float = 0.20
    eval_episodes_per_map: int = 4
    eval_dynamic_obstacles: int = 1
    architecture: str = "tiny_mlp"
    mlp_hidden_dim: int = 32
    mlp_epochs: int = 90
    mlp_lr: float = 8e-3
    mlp_batch_size: int = 256


def _blend_action(teacher: Action, model: Action, teacher_alpha: float) -> Action:
    throttle = teacher_alpha * teacher.throttle + (1.0 - teacher_alpha) * model.throttle
    steering = teacher_alpha * teacher.steering + (1.0 - teacher_alpha) * model.steering
    return Action(
        throttle=float(np.clip(throttle, -1.0, 1.0)),
        steering=float(np.clip(steering, -1.0, 1.0)),
    )


def _collect_supervised_rows(
    policy: object | None,
    cfg: TrainAnyoneConfig,
    episodes_per_map: int,
    teacher_alpha: float,
    seed_offset: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(cfg.seed + seed_offset)
    feature_rows: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []

    env = DriveSimEnv(
        EnvConfig(
            map_name=cfg.maps[0],
            max_steps=cfg.max_steps,
            auto_expand=False,
            dynamic_obstacle_count=cfg.dynamic_obstacle_count,
            seed=cfg.seed + seed_offset,
        )
    )

    for map_name in cfg.maps:
        env.set_map(map_name)
        for _ in range(episodes_per_map):
            env.reset(seed=int(rng.integers(0, 2**31 - 1)))
            obs = env.randomize_episode(rng=rng, min_goal_distance=cfg.min_goal_distance)
            done = False
            while not done:
                teacher_action = env.autopilot_action(env.sim.get_state())
                model_action = policy.act(obs) if policy is not None else teacher_action  # type: ignore[attr-defined]
                action = _blend_action(teacher_action, model_action, teacher_alpha)
                feature_rows.append(features_from_observation(obs).astype(np.float32))
                action_rows.append(np.asarray([teacher_action.throttle, teacher_action.steering], dtype=np.float32))
                obs, _, done, _ = env.step(action)

    x = np.vstack(feature_rows)
    y = np.vstack(action_rows)
    return x, y


def _fit_tiny_mlp(x: np.ndarray, y: np.ndarray, cfg: TrainAnyoneConfig, seed_offset: int) -> TinyMLPPolicyModel:
    rng = np.random.default_rng(cfg.seed + seed_offset)
    mean = x.mean(axis=0).astype(np.float32)
    std = (x.std(axis=0) + 1e-6).astype(np.float32)
    xn = ((x - mean) / std).astype(np.float32)
    yn = y.astype(np.float32)

    n, dim = xn.shape
    hidden = cfg.mlp_hidden_dim
    w1 = (rng.normal(0.0, 0.25, size=(dim, hidden)) / np.sqrt(max(1, dim))).astype(np.float32)
    b1 = np.zeros(hidden, dtype=np.float32)
    w2 = (rng.normal(0.0, 0.25, size=(hidden, 2)) / np.sqrt(max(1, hidden))).astype(np.float32)
    b2 = np.zeros(2, dtype=np.float32)
    batch_size = max(8, min(cfg.mlp_batch_size, n))

    for _ in range(cfg.mlp_epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            xb = xn[idx]
            yb = yn[idx]

            h = np.tanh(xb @ w1 + b1)
            pred = h @ w2 + b2
            err = pred - yb

            scale = np.float32(2.0 / max(1, xb.shape[0]))
            grad_pred = err * scale
            grad_w2 = h.T @ grad_pred
            grad_b2 = grad_pred.sum(axis=0)

            grad_h = grad_pred @ w2.T
            grad_z1 = grad_h * (1.0 - h * h)
            grad_w1 = xb.T @ grad_z1
            grad_b1 = grad_z1.sum(axis=0)

            # Small L2 stabilizer.
            grad_w2 += np.float32(cfg.l2_reg) * w2
            grad_w1 += np.float32(cfg.l2_reg) * w1

            w2 -= np.float32(cfg.mlp_lr) * grad_w2
            b2 -= np.float32(cfg.mlp_lr) * grad_b2
            w1 -= np.float32(cfg.mlp_lr) * grad_w1
            b1 -= np.float32(cfg.mlp_lr) * grad_b1

    return TinyMLPPolicyModel(
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
        feature_mean=mean,
        feature_std=std,
    )


def _fit_model(x: np.ndarray, y: np.ndarray, cfg: TrainAnyoneConfig, seed_offset: int) -> object:
    if cfg.architecture == "linear":
        return fit_linear_policy(x, y, l2_reg=cfg.l2_reg)
    return _fit_tiny_mlp(x, y, cfg, seed_offset=seed_offset)


def _save_model(model: object, path: str) -> None:
    if isinstance(model, LinearPolicy):
        model.save(path)
        return
    if isinstance(model, TinyMLPPolicyModel):
        model.save(path)
        return
    raise TypeError(f"unsupported model type: {type(model)!r}")


def train_anyone(cfg: TrainAnyoneConfig) -> tuple[object, dict[str, object]]:
    base_x, base_y = _collect_supervised_rows(
        policy=None,
        cfg=cfg,
        episodes_per_map=cfg.bc_episodes_per_map,
        teacher_alpha=1.0,
        seed_offset=0,
    )
    x_parts = [base_x]
    y_parts = [base_y]

    policy = _fit_model(base_x, base_y, cfg, seed_offset=10)
    print(
        f"[bc] rows={base_x.shape[0]} maps={len(cfg.maps)} "
        f"episodes_per_map={cfg.bc_episodes_per_map}",
        flush=True,
    )

    for round_idx in range(cfg.dagger_rounds):
        if cfg.dagger_rounds <= 1:
            alpha = cfg.teacher_mix_end
        else:
            t = round_idx / max(1, cfg.dagger_rounds - 1)
            alpha = cfg.teacher_mix_start + (cfg.teacher_mix_end - cfg.teacher_mix_start) * t

        round_x, round_y = _collect_supervised_rows(
            policy=policy,
            cfg=cfg,
            episodes_per_map=cfg.dagger_episodes_per_map,
            teacher_alpha=float(alpha),
            seed_offset=1000 + round_idx * 73,
        )
        x_parts.append(round_x)
        y_parts.append(round_y)
        x_all = np.vstack(x_parts)
        y_all = np.vstack(y_parts)
        policy = _fit_model(x_all, y_all, cfg, seed_offset=400 + round_idx * 31)
        print(
            f"[dagger {round_idx + 1:02d}/{cfg.dagger_rounds}] "
            f"alpha={alpha:.2f} new_rows={round_x.shape[0]} total_rows={x_all.shape[0]}",
            flush=True,
        )

    _save_model(policy, cfg.output_path)
    print(f"saved policy -> {cfg.output_path}", flush=True)

    summary = run_eval(
        EvalConfig(
            maps=cfg.maps,
            episodes_per_map=cfg.eval_episodes_per_map,
            max_steps=cfg.max_steps,
            seed=cfg.seed + 404,
            policy_mode="assistant",
            model_path=cfg.output_path,
            dynamic_obstacle_count=cfg.eval_dynamic_obstacles,
        )
    )
    print(
        "[eval] "
        f"episodes={int(summary['episodes'])} "
        f"success={summary['success_rate']:.1%} "
        f"collision={summary['collision_rate']:.1%} "
        f"avg_dist={summary['avg_distance_to_goal']:.1f}",
        flush=True,
    )
    return policy, summary


def _apply_profile(profile: str, cfg: TrainAnyoneConfig) -> TrainAnyoneConfig:
    if profile == "quick":
        cfg.bc_episodes_per_map = 3
        cfg.dagger_rounds = 1
        cfg.dagger_episodes_per_map = 1
        cfg.eval_episodes_per_map = 2
    elif profile == "strong":
        cfg.bc_episodes_per_map = 24
        cfg.dagger_rounds = 2
        cfg.dagger_episodes_per_map = 12
        cfg.eval_episodes_per_map = 6
        cfg.mlp_epochs = 140
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an assistant agent with robust defaults (BC + DAgger) for first-time users."
    )
    parser.add_argument("--maps", default="default,maze,blocks", help="Comma-separated training maps")
    parser.add_argument("--seed", type=int, default=21, help="Base seed")
    parser.add_argument("--output", default="models/assist_policy.npz", help="Output model path")
    parser.add_argument("--max-steps", type=int, default=600, help="Max steps per episode")
    parser.add_argument("--min-goal-distance", type=float, default=180.0, help="Minimum randomized start->goal distance")
    parser.add_argument("--dynamic-obstacles", type=int, default=0, help="Training dynamic obstacle count")
    parser.add_argument("--eval-dynamic-obstacles", type=int, default=1, help="Eval dynamic obstacle count")
    parser.add_argument("--l2", type=float, default=5e-3, help="L2 regularization strength")
    parser.add_argument("--arch", choices=["linear", "tiny_mlp"], default="tiny_mlp", help="Policy architecture")
    parser.add_argument("--mlp-hidden", type=int, default=32, help="Tiny MLP hidden units")
    parser.add_argument("--mlp-epochs", type=int, default=90, help="Tiny MLP epochs")
    parser.add_argument("--mlp-lr", type=float, default=8e-3, help="Tiny MLP learning rate")
    parser.add_argument("--mlp-batch", type=int, default=256, help="Tiny MLP batch size")
    parser.add_argument("--bc-episodes", type=int, default=4, help="Behavior-cloning episodes per map")
    parser.add_argument("--dagger-rounds", type=int, default=1, help="DAgger refinement rounds")
    parser.add_argument("--dagger-episodes", type=int, default=2, help="DAgger episodes per map per round")
    parser.add_argument("--eval-episodes", type=int, default=4, help="Evaluation episodes per map")
    parser.add_argument(
        "--profile",
        choices=["quick", "standard", "strong"],
        default="standard",
        help="Preset that adjusts episode counts for speed vs quality",
    )
    args = parser.parse_args()

    maps = [m.strip() for m in args.maps.split(",") if m.strip()]
    cfg = TrainAnyoneConfig(
        maps=maps,
        seed=args.seed,
        output_path=args.output,
        max_steps=args.max_steps,
        min_goal_distance=args.min_goal_distance,
        dynamic_obstacle_count=args.dynamic_obstacles,
        l2_reg=args.l2,
        architecture=args.arch,
        mlp_hidden_dim=args.mlp_hidden,
        mlp_epochs=args.mlp_epochs,
        mlp_lr=args.mlp_lr,
        mlp_batch_size=args.mlp_batch,
        bc_episodes_per_map=args.bc_episodes,
        dagger_rounds=args.dagger_rounds,
        dagger_episodes_per_map=args.dagger_episodes,
        eval_episodes_per_map=args.eval_episodes,
        eval_dynamic_obstacles=args.eval_dynamic_obstacles,
    )
    if args.profile != "standard":
        cfg = _apply_profile(args.profile, cfg)
    train_anyone(cfg)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from dataclasses import dataclass
import time

import numpy as np

from drivesim.ml.env import DriveSimEnv, EnvConfig
from drivesim.ml.policy import LinearPolicy, features_from_observation


@dataclass
class AutoTrainConfig:
    iterations: int = 16
    population: int = 24
    elite_fraction: float = 0.25
    episodes_per_candidate: int = 3
    map_name: str = "default"
    max_steps: int = 800
    seed: int = 7
    output_path: str = "models/assist_policy.npz"
    show_candidate_progress: bool = True


def _vector_to_policy(vec: np.ndarray, feature_dim: int) -> LinearPolicy:
    weight_size = feature_dim * 2
    weights = vec[:weight_size].reshape(feature_dim, 2).astype(np.float32)
    bias = vec[weight_size:].astype(np.float32)
    mean = np.zeros(feature_dim, dtype=np.float32)
    std = np.ones(feature_dim, dtype=np.float32)
    return LinearPolicy(weights=weights, bias=bias, feature_mean=mean, feature_std=std)


def _evaluate_policy(policy: LinearPolicy, cfg: AutoTrainConfig) -> tuple[float, float, float]:
    env = DriveSimEnv(
        EnvConfig(
            map_name=cfg.map_name,
            max_steps=cfg.max_steps,
            auto_expand=False,
        )
    )

    scores: list[float] = []
    success_count = 0
    collision_count = 0

    for _ in range(cfg.episodes_per_candidate):
        obs = env.reset()
        done = False
        total_reward = 0.0
        info = {"distance_to_goal": 0.0}
        while not done:
            action = policy.act(obs)
            obs, reward, done, info = env.step(action)
            total_reward += float(reward)

        distance = float(info["distance_to_goal"])
        success = distance < 18.0 and not bool(obs["collided"])
        collision = bool(obs["collided"])
        if success:
            success_count += 1
        if collision:
            collision_count += 1

        # Shape objective strongly toward safe goal completion.
        score = total_reward - 0.04 * distance
        if success:
            score += 180.0
        if collision:
            score -= 70.0
        scores.append(score)

    avg_score = float(np.mean(scores))
    success_rate = success_count / cfg.episodes_per_candidate
    collision_rate = collision_count / cfg.episodes_per_candidate
    return avg_score, success_rate, collision_rate


def train_auto(cfg: AutoTrainConfig) -> LinearPolicy:
    rng = np.random.default_rng(cfg.seed)
    probe_env = DriveSimEnv(EnvConfig(map_name=cfg.map_name, max_steps=cfg.max_steps))
    feature_dim = len(features_from_observation(probe_env.reset()))
    param_dim = feature_dim * 2 + 2

    mean = np.zeros(param_dim, dtype=np.float32)
    mean[-2] = 0.2
    std = np.ones(param_dim, dtype=np.float32) * 0.7
    std[-2] = 0.35
    std[-1] = 0.35

    elite_count = max(2, int(cfg.population * cfg.elite_fraction))
    best_score = -1e18
    best_vec = mean.copy()
    best_success = 0.0
    train_start = time.time()
    total_candidates = cfg.iterations * cfg.population

    for i in range(cfg.iterations):
        candidates = mean + std * rng.normal(size=(cfg.population, param_dim)).astype(np.float32)
        scored: list[tuple[float, float, float, np.ndarray]] = []
        iter_start = time.time()
        for j, vec in enumerate(candidates):
            policy = _vector_to_policy(vec, feature_dim)
            score, success_rate, collision_rate = _evaluate_policy(policy, cfg)
            scored.append((score, success_rate, collision_rate, vec))
            if cfg.show_candidate_progress:
                done = i * cfg.population + j + 1
                elapsed = max(1e-6, time.time() - train_start)
                rate = done / elapsed
                remaining = max(0.0, (total_candidates - done) / max(1e-6, rate))
                msg = (
                    f"\rtraining {done:4d}/{total_candidates} "
                    f"iter={i + 1:02d}/{cfg.iterations} cand={j + 1:02d}/{cfg.population} "
                    f"score={score:8.2f} success={success_rate:5.1%} "
                    f"collision={collision_rate:5.1%} eta={remaining:6.1f}s"
                )
                print(msg, end="", flush=True)

        scored.sort(key=lambda x: x[0], reverse=True)
        elites = scored[:elite_count]
        elite_vectors = np.stack([e[3] for e in elites], axis=0)
        mean = elite_vectors.mean(axis=0)
        std = elite_vectors.std(axis=0) + 0.03

        top_score, top_success, top_collision, top_vec = elites[0]
        if top_score > best_score:
            best_score = top_score
            best_vec = top_vec.copy()
            best_success = top_success

        if cfg.show_candidate_progress:
            print("", flush=True)
        iter_time = time.time() - iter_start
        print(
            f"[iter {i + 1:02d}/{cfg.iterations}] "
            f"best_score={top_score:8.2f} success={top_success:5.1%} "
            f"collision={top_collision:5.1%} iter_time={iter_time:5.1f}s",
            flush=True,
        )

    best_policy = _vector_to_policy(best_vec, feature_dim)
    best_policy.save(cfg.output_path)
    print(
        f"saved auto-trained policy -> {cfg.output_path} "
        f"(best_score={best_score:.2f}, success={best_success:.1%})",
        flush=True,
    )
    return best_policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-train assistant policy via many headless episodes.")
    parser.add_argument("--iterations", type=int, default=16, help="Optimization iterations")
    parser.add_argument("--population", type=int, default=24, help="Candidates per iteration")
    parser.add_argument("--elite-fraction", type=float, default=0.25, help="Top candidate fraction used to update")
    parser.add_argument("--episodes", type=int, default=3, help="Episodes per candidate evaluation")
    parser.add_argument("--map", default="default", help="Map preset name")
    parser.add_argument("--max-steps", type=int, default=800, help="Max steps per episode")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--output", default="models/assist_policy.npz", help="Output model path")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable per-candidate live progress and show only per-iteration summaries",
    )
    args = parser.parse_args()

    cfg = AutoTrainConfig(
        iterations=args.iterations,
        population=args.population,
        elite_fraction=args.elite_fraction,
        episodes_per_candidate=args.episodes,
        map_name=args.map,
        max_steps=args.max_steps,
        seed=args.seed,
        output_path=args.output,
        show_candidate_progress=not args.quiet,
    )
    train_auto(cfg)


if __name__ == "__main__":
    main()

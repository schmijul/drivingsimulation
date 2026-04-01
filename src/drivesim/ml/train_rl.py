from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path

from drivesim.ml.curriculum import CurriculumStage, resolve_curriculum, single_stage_curriculum
from drivesim.ml.env import DriveSimGymEnv, EnvConfig
from drivesim.ml.eval import EvalConfig, run_eval
from drivesim.ml.experiment import (
    append_experiment_record,
    build_experiment_record,
    default_experiment_history_path,
)
from drivesim.ml.rl_utils import PPO, require_sb3

try:
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
except ImportError:  # pragma: no cover - optional dependency
    BaseCallback = object
    CallbackList = None
    CheckpointCallback = None
    DummyVecEnv = None
    VecMonitor = None


@dataclass
class RLTrainConfig:
    maps: list[str]
    total_timesteps: int = 4096
    num_envs: int = 1
    max_steps: int = 600
    seed: int = 21
    algo: str = "ppo"
    output_dir: str = "models/rl"
    run_name: str = ""
    observation_mode: str = "flat"
    dynamic_obstacle_count: int = 0
    mapping_mode: str = "ground_truth"
    min_goal_distance: float = 160.0
    curriculum: str = "standard"
    learning_rate: float = 3e-4
    gamma: float = 0.99
    ppo_n_steps: int = 256
    ppo_batch_size: int = 64
    policy_hidden_sizes: tuple[int, ...] = (64, 64)
    eval_episodes_per_map: int = 2
    track_run: bool = False
    experiment_history_path: str = default_experiment_history_path()


class CurriculumProgressCallback(BaseCallback):
    def __init__(self, stages: tuple[CurriculumStage, ...], total_timesteps: int):
        super().__init__()
        self.stages = stages
        self.total_timesteps = max(1, total_timesteps)
        self.current_stage_idx = -1

    def _set_stage(self, stage_idx: int) -> None:
        if stage_idx == self.current_stage_idx:
            return
        stage = self.stages[stage_idx]
        self.training_env.env_method("apply_curriculum_stage", stage.to_dict())
        self.current_stage_idx = stage_idx
        print(
            f"[curriculum] stage={stage.name} maps={','.join(stage.maps)} "
            f"dynamic={stage.dynamic_obstacle_count} mapping={stage.mapping_mode}",
            flush=True,
        )

    def _on_training_start(self) -> None:
        self._set_stage(0)

    def _on_step(self) -> bool:
        stage_span = max(1, self.total_timesteps // max(1, len(self.stages)))
        stage_idx = min(len(self.stages) - 1, self.num_timesteps // stage_span)
        self._set_stage(stage_idx)
        return True


def default_rl_run_name(cfg: RLTrainConfig, now: datetime | None = None) -> str:
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    scope = cfg.curriculum or "-".join(cfg.maps)
    safe_scope = scope.replace(",", "-")
    return f"{cfg.algo}_{safe_scope}_seed{cfg.seed}_{stamp}"


def _make_env_factory(
    *,
    rank: int,
    cfg: RLTrainConfig,
    allowed_maps: list[str],
    initial_stage: CurriculumStage,
):
    def _factory() -> DriveSimGymEnv:
        env = DriveSimGymEnv(
            EnvConfig(
                map_name=allowed_maps[0],
                max_steps=cfg.max_steps,
                auto_expand=False,
                dynamic_obstacle_count=initial_stage.dynamic_obstacle_count,
                seed=cfg.seed + rank,
                mapping_mode=initial_stage.mapping_mode,
            ),
            observation_mode=cfg.observation_mode,
            map_names=allowed_maps,
            randomize_on_reset=True,
            min_goal_distance=initial_stage.min_goal_distance,
        )
        env.apply_curriculum_stage(initial_stage.to_dict())
        return env

    return _factory


def train_rl(cfg: RLTrainConfig) -> dict[str, object]:
    require_sb3()
    if cfg.algo != "ppo":
        raise ValueError(f"unsupported RL algorithm: {cfg.algo!r}")
    if DummyVecEnv is None or VecMonitor is None or CheckpointCallback is None or CallbackList is None:
        raise ImportError("stable-baselines3 callback/vector env support is unavailable")

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
    allowed_maps = sorted({map_name for stage in curriculum.stages for map_name in stage.maps})
    run_name = cfg.run_name or default_rl_run_name(cfg)
    run_dir = Path(cfg.output_dir) / run_name
    checkpoint_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    config_payload = {
        **asdict(cfg),
        "curriculum": curriculum.name,
        "policy_hidden_sizes": list(cfg.policy_hidden_sizes),
        "allowed_maps": allowed_maps,
        "stages": [stage.to_dict() for stage in curriculum.stages],
    }
    (run_dir / "train_config.json").write_text(json.dumps(config_payload, indent=2) + "\n", encoding="utf-8")

    env = VecMonitor(
        DummyVecEnv(
            [
                _make_env_factory(
                    rank=rank,
                    cfg=cfg,
                    allowed_maps=allowed_maps,
                    initial_stage=curriculum.stages[0],
                )
                for rank in range(max(1, cfg.num_envs))
            ]
        )
    )

    checkpoint_steps = max(cfg.ppo_n_steps, cfg.total_timesteps // 4)
    callback = CallbackList(
        [
            CurriculumProgressCallback(curriculum.stages, cfg.total_timesteps),
            CheckpointCallback(
                save_freq=max(1, checkpoint_steps // max(1, cfg.num_envs)),
                save_path=str(checkpoint_dir),
                name_prefix=f"{cfg.algo}_checkpoint",
            ),
        ]
    )

    assert PPO is not None
    model = PPO(
        "MlpPolicy",
        env,
        seed=cfg.seed,
        learning_rate=cfg.learning_rate,
        gamma=cfg.gamma,
        n_steps=cfg.ppo_n_steps,
        batch_size=cfg.ppo_batch_size,
        policy_kwargs={"net_arch": list(cfg.policy_hidden_sizes)},
        verbose=1,
    )
    model.learn(total_timesteps=cfg.total_timesteps, callback=callback)

    model_path = run_dir / "model"
    model.save(str(model_path))
    env.close()

    final_model_path = str(model_path.with_suffix(".zip"))
    eval_path = run_dir / "eval.json"
    eval_cfg = EvalConfig(
        maps=cfg.maps,
        episodes_per_map=cfg.eval_episodes_per_map,
        max_steps=cfg.max_steps,
        seed=cfg.seed + 404,
        policy_mode="rl",
        model_path=final_model_path,
        dynamic_obstacle_count=cfg.dynamic_obstacle_count,
        mapping_mode=cfg.mapping_mode,
        min_goal_distance=cfg.min_goal_distance,
        curriculum=cfg.curriculum,
        rl_algorithm=cfg.algo,
        json_out=str(eval_path),
        track_run=False,
        experiment_history_path=cfg.experiment_history_path,
    )
    eval_summary = run_eval(eval_cfg)

    if cfg.track_run:
        row = build_experiment_record(
            kind="train_rl",
            name=run_name,
            algo=cfg.algo,
            policy_mode="rl",
            model_path=final_model_path,
            seed=cfg.seed,
            maps=cfg.maps,
            curriculum=curriculum.name,
            summary=eval_summary,
            config=config_payload,
            report_path=str(eval_path),
        )
        append_experiment_record(cfg.experiment_history_path, row)

    print(
        f"saved rl model -> {final_model_path} "
        f"(success={eval_summary['success_rate']:.1%}, "
        f"collision={eval_summary['collision_rate']:.1%})",
        flush=True,
    )
    return {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "model_path": final_model_path,
        "eval_summary": eval_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a PPO policy on DriveSim with Gym/SB3 compatibility.")
    parser.add_argument("--maps", default="default,maze,blocks", help="Comma-separated map list")
    parser.add_argument("--timesteps", type=int, default=4096, help="Total RL training timesteps")
    parser.add_argument("--num-envs", type=int, default=1, help="Parallel environments")
    parser.add_argument("--max-steps", type=int, default=600, help="Max steps per episode")
    parser.add_argument("--seed", type=int, default=21, help="Training seed")
    parser.add_argument("--algo", choices=["ppo"], default="ppo", help="RL algorithm")
    parser.add_argument("--output-dir", default="models/rl", help="Directory for run outputs")
    parser.add_argument("--run-name", default="", help="Optional fixed run name")
    parser.add_argument("--observation-mode", choices=["flat"], default="flat", help="Observation mode")
    parser.add_argument("--dynamic-obstacles", type=int, default=0, help="Base dynamic obstacle count without curriculum")
    parser.add_argument(
        "--mapping-mode",
        choices=["ground_truth", "sensor_driven"],
        default="ground_truth",
        help="Base mapping mode without curriculum",
    )
    parser.add_argument("--min-goal-distance", type=float, default=160.0, help="Base min goal distance")
    parser.add_argument(
        "--curriculum",
        choices=["", "easy", "standard", "robust"],
        default="standard",
        help="Optional curriculum preset",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="PPO learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--ppo-n-steps", type=int, default=256, help="PPO rollout steps per environment")
    parser.add_argument("--ppo-batch-size", type=int, default=64, help="PPO batch size")
    parser.add_argument("--hidden-sizes", default="64,64", help="Comma-separated MLP hidden sizes")
    parser.add_argument("--eval-episodes", type=int, default=2, help="Evaluation episodes per map after training")
    parser.add_argument(
        "--no-track-run",
        action="store_true",
        help="Disable experiment history tracking for this CLI invocation",
    )
    args = parser.parse_args()

    maps = [map_name.strip() for map_name in args.maps.split(",") if map_name.strip()]
    hidden_sizes = tuple(int(part.strip()) for part in args.hidden_sizes.split(",") if part.strip())
    cfg = RLTrainConfig(
        maps=maps,
        total_timesteps=args.timesteps,
        num_envs=args.num_envs,
        max_steps=args.max_steps,
        seed=args.seed,
        algo=args.algo,
        output_dir=args.output_dir,
        run_name=args.run_name,
        observation_mode=args.observation_mode,
        dynamic_obstacle_count=args.dynamic_obstacles,
        mapping_mode=args.mapping_mode,
        min_goal_distance=args.min_goal_distance,
        curriculum=args.curriculum,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        ppo_n_steps=args.ppo_n_steps,
        ppo_batch_size=args.ppo_batch_size,
        policy_hidden_sizes=hidden_sizes,
        eval_episodes_per_map=args.eval_episodes,
        track_run=not args.no_track_run,
    )
    train_rl(cfg)


if __name__ == "__main__":
    main()

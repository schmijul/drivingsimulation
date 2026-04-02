from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Protocol

import numpy as np

from drivesim.core.scenario import build_world
from drivesim.core.types import Action
from drivesim.ml.policy import LinearPolicy, features_for_dim
from drivesim.ml.rl_utils import load_rl_model


class PolicyModel(Protocol):
    model_name: str

    def act(self, observation: dict) -> Action:
        ...


@dataclass
class LinearPolicyModel:
    policy: LinearPolicy
    model_name: str = "linear"

    def act(self, observation: dict) -> Action:
        return self.policy.act(observation)


@dataclass
class TinyMLPPolicyModel:
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    feature_mean: np.ndarray
    feature_std: np.ndarray
    model_name: str = "tiny_mlp"

    def act(self, observation: dict) -> Action:
        x = features_for_dim(observation, int(self.feature_mean.shape[0])).astype(np.float32)
        x = (x - self.feature_mean) / self.feature_std
        h = np.tanh(x @ self.w1 + self.b1)
        out = h @ self.w2 + self.b2
        throttle = float(np.clip(out[0], -1.0, 1.0))
        steering = float(np.clip(out[1], -1.0, 1.0))
        return Action(throttle=throttle, steering=steering)

    def save(self, path: str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            target,
            model_type=self.model_name,
            w1=self.w1,
            b1=self.b1,
            w2=self.w2,
            b2=self.b2,
            feature_mean=self.feature_mean,
            feature_std=self.feature_std,
        )

    @classmethod
    def load(cls, path: str) -> "TinyMLPPolicyModel":
        data = np.load(path)
        return cls(
            w1=data["w1"],
            b1=data["b1"],
            w2=data["w2"],
            b2=data["b2"],
            feature_mean=data["feature_mean"],
            feature_std=data["feature_std"],
        )


@dataclass
class PPOPolicyModel:
    model: object
    allowed_maps: tuple[str, ...]
    grid_shape: tuple[int, int]
    model_name: str = "ppo"

    def _flat_observation(self, observation: dict) -> np.ndarray:
        map_name = str(observation.get("map_name", ""))
        if map_name and self.allowed_maps and map_name not in self.allowed_maps:
            raise ValueError(f"map {map_name!r} is outside PPO training maps {self.allowed_maps!r}")

        pose = np.asarray(observation["pose"], dtype=np.float32)
        goal = np.asarray(observation["goal"], dtype=np.float32)
        grid = np.asarray(observation["grid"], dtype=np.float32)
        lidar = np.asarray(observation["lidar"], dtype=np.float32)

        target_rows, target_cols = self.grid_shape
        rows, cols = grid.shape
        if rows > target_rows or cols > target_cols:
            raise ValueError(
                f"observation grid {grid.shape} exceeds PPO input grid shape {self.grid_shape}; "
                "use a checkpoint trained on this map family"
            )

        padded_grid = np.zeros(self.grid_shape, dtype=np.float32)
        padded_grid[:rows, :cols] = grid
        collided = np.array([float(bool(observation.get("collided", False)))], dtype=np.float32)
        return np.concatenate([pose, goal, padded_grid.ravel(), lidar, collided]).astype(np.float32, copy=False)

    def act(self, observation: dict) -> Action:
        flat_obs = self._flat_observation(observation)
        action, _ = self.model.predict(flat_obs, deterministic=True)  # type: ignore[attr-defined]
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if arr.size < 2:
            raise ValueError("PPO model returned an invalid action shape")
        return Action(
            throttle=float(np.clip(arr[0], -1.0, 1.0)),
            steering=float(np.clip(arr[1], -1.0, 1.0)),
        )


def list_model_architectures() -> list[str]:
    return ["linear", "tiny_mlp"]


def _grid_shape_for_maps(map_names: list[str]) -> tuple[int, int]:
    max_width = 0.0
    max_height = 0.0
    resolution = 8.0
    for map_name in map_names:
        world = build_world(map_name)
        max_width = max(max_width, world.width)
        max_height = max(max_height, world.height)
    return int(max_height // resolution) + 1, int(max_width // resolution) + 1


def _load_ppo_policy_model(path: str) -> PPOPolicyModel:
    model_path = Path(path)
    config_path = model_path.with_name("train_config.json")
    allowed_maps: list[str] = ["default"]
    grid_shape = (57, 101)
    if config_path.exists():
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        raw_maps = payload.get("allowed_maps", ["default"])
        allowed_maps = [str(map_name) for map_name in raw_maps]
        raw_grid_shape = payload.get("target_grid_shape")
        if isinstance(raw_grid_shape, list) and len(raw_grid_shape) == 2:
            grid_shape = (int(raw_grid_shape[0]), int(raw_grid_shape[1]))
        else:
            grid_shape = _grid_shape_for_maps(allowed_maps)
    model = load_rl_model(path, algorithm="ppo")
    return PPOPolicyModel(model=model, allowed_maps=tuple(allowed_maps), grid_shape=grid_shape)


def load_policy_model(path: str) -> PolicyModel:
    if Path(path).suffix == ".zip":
        return _load_ppo_policy_model(path)

    data = np.load(path)
    model_type = "linear"
    if "model_type" in data:
        raw = data["model_type"]
        model_type = str(raw.item() if np.ndim(raw) == 0 else raw)

    if model_type == "tiny_mlp":
        return TinyMLPPolicyModel(
            w1=data["w1"],
            b1=data["b1"],
            w2=data["w2"],
            b2=data["b2"],
            feature_mean=data["feature_mean"],
            feature_std=data["feature_std"],
        )

    policy = LinearPolicy(
        weights=data["weights"],
        bias=data["bias"],
        feature_mean=data["feature_mean"],
        feature_std=data["feature_std"],
    )
    return LinearPolicyModel(policy=policy)

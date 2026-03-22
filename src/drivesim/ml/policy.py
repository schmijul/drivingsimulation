from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from drivesim.core.types import Action


@dataclass
class LinearPolicy:
    weights: np.ndarray
    bias: np.ndarray
    feature_mean: np.ndarray
    feature_std: np.ndarray

    def predict(self, features: np.ndarray) -> np.ndarray:
        normalized = (features - self.feature_mean) / self.feature_std
        return normalized @ self.weights + self.bias

    def act(self, observation: dict) -> Action:
        features = features_from_observation(observation)
        pred = self.predict(features)
        throttle = float(np.clip(pred[0], -1.0, 1.0))
        steering = float(np.clip(pred[1], -1.0, 1.0))
        return Action(throttle=throttle, steering=steering)

    def save(self, path: str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            target,
            weights=self.weights,
            bias=self.bias,
            feature_mean=self.feature_mean,
            feature_std=self.feature_std,
        )

    @classmethod
    def load(cls, path: str) -> "LinearPolicy":
        data = np.load(path)
        return cls(
            weights=data["weights"],
            bias=data["bias"],
            feature_mean=data["feature_mean"],
            feature_std=data["feature_std"],
        )


def features_from_observation(observation: dict) -> np.ndarray:
    pose = np.asarray(observation["pose"], dtype=np.float32)
    goal = np.asarray(observation["goal"], dtype=np.float32)
    lidar = np.asarray(observation["lidar"], dtype=np.float32)
    samples = lidar[:: max(1, len(lidar) // 8)]
    goal_dx = goal[0] - pose[0]
    goal_dy = goal[1] - pose[1]
    feats = np.concatenate(
        [
            np.array([pose[3], goal_dx, goal_dy], dtype=np.float32),
            samples.astype(np.float32),
        ]
    )
    return feats


def fit_linear_policy(features: np.ndarray, actions: np.ndarray, l2_reg: float = 1e-2) -> LinearPolicy:
    mean = features.mean(axis=0)
    std = features.std(axis=0) + 1e-6
    x = (features - mean) / std
    n_features = x.shape[1]
    eye = np.eye(n_features, dtype=np.float32)
    xtx = x.T @ x + l2_reg * eye
    w = np.linalg.solve(xtx, x.T @ actions)
    b = actions.mean(axis=0)
    return LinearPolicy(weights=w, bias=b, feature_mean=mean, feature_std=std)

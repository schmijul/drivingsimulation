from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from drivesim.core.types import Action
from drivesim.ml.models import LinearPolicyModel
from drivesim.ml.policy import LinearPolicy, features_from_observation


@dataclass
class LiveTrainConfig:
    population: int = 8
    elite_fraction: float = 0.25
    seed: int = 9


class LivePolicyTrainer:
    """Online visual trainer for the linear assistant policy."""

    def __init__(self, feature_dim: int, config: LiveTrainConfig | None = None):
        self.config = config or LiveTrainConfig()
        self.feature_dim = feature_dim
        self.param_dim = feature_dim * 2 + 2
        self.rng = np.random.default_rng(self.config.seed)

        self.mean = np.zeros(self.param_dim, dtype=np.float32)
        self.std = np.ones(self.param_dim, dtype=np.float32) * 0.6
        self.mean[-2] = 0.2
        self.std[-2] = 0.35
        self.std[-1] = 0.35

        self.iteration = 1
        self.candidate_idx = 0
        self.episode_reward = 0.0
        self.eval_rows: list[tuple[float, np.ndarray]] = []
        self.best_score = -1e18
        self.best_vec = self.mean.copy()
        self.last_score = 0.0
        self.current_vec = self._sample_candidate()
        self.current_model = self._vector_to_model(self.current_vec)

    def _sample_candidate(self) -> np.ndarray:
        return self.mean + self.std * self.rng.normal(size=self.param_dim).astype(np.float32)

    def _vector_to_model(self, vec: np.ndarray) -> LinearPolicyModel:
        weight_size = self.feature_dim * 2
        weights = vec[:weight_size].reshape(self.feature_dim, 2).astype(np.float32)
        bias = vec[weight_size:].astype(np.float32)
        policy = LinearPolicy(
            weights=weights,
            bias=bias,
            feature_mean=np.zeros(self.feature_dim, dtype=np.float32),
            feature_std=np.ones(self.feature_dim, dtype=np.float32),
        )
        return LinearPolicyModel(policy=policy)

    def act(self, observation: dict) -> Action:
        return self.current_model.act(observation)

    def observe(self, reward: float, done: bool, observation: dict, info: dict) -> bool:
        self.episode_reward += float(reward)
        if not done:
            return False

        distance = float(info.get("distance_to_goal", 0.0))
        collided = bool(observation.get("collided", False))
        success = distance < 18.0 and not collided

        score = self.episode_reward - 0.04 * distance
        if success:
            score += 180.0
        if collided:
            score -= 70.0

        self.last_score = score
        if score > self.best_score:
            self.best_score = score
            self.best_vec = self.current_vec.copy()

        self.eval_rows.append((score, self.current_vec.copy()))
        self.episode_reward = 0.0
        self.candidate_idx += 1

        if self.candidate_idx >= self.config.population:
            self._finish_iteration()

        self.current_vec = self._sample_candidate()
        self.current_model = self._vector_to_model(self.current_vec)
        return True

    def _finish_iteration(self) -> None:
        rows = sorted(self.eval_rows, key=lambda r: r[0], reverse=True)
        elite_count = max(2, int(self.config.population * self.config.elite_fraction))
        elites = np.stack([r[1] for r in rows[:elite_count]], axis=0)
        self.mean = elites.mean(axis=0)
        self.std = elites.std(axis=0) + 0.03
        self.eval_rows = []
        self.candidate_idx = 0
        self.iteration += 1

    def status_label(self) -> str:
        return (
            "train-live "
            f"| iter={self.iteration} cand={self.candidate_idx + 1}/{self.config.population} "
            f"| last={self.last_score:6.1f} best={self.best_score:6.1f}"
        )

    def best_model(self) -> LinearPolicyModel:
        return self._vector_to_model(self.best_vec)

    @staticmethod
    def feature_dim_from_observation(observation: dict) -> int:
        return len(features_from_observation(observation))

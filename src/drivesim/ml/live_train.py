from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from drivesim.core.types import Action
from drivesim.ml.models import LinearPolicyModel
from drivesim.ml.policy import LinearPolicy, features_from_observation, fit_linear_policy


@dataclass
class LiveTrainConfig:
    max_buffer: int = 12000
    fit_every_steps: int = 40
    min_fit_samples: int = 96
    teacher_blend_start: float = 0.95
    teacher_blend_end: float = 0.20
    teacher_blend_decay_steps: int = 1200
    seed: int = 9


class LivePolicyTrainer:
    """Online visual trainer with teacher-guided imitation for stability."""

    def __init__(self, feature_dim: int, config: LiveTrainConfig | None = None):
        self.config = config or LiveTrainConfig()
        self.feature_dim = feature_dim
        self.param_dim = feature_dim * 2 + 2
        self.rng = np.random.default_rng(self.config.seed)
        self.steps = 0
        self.fit_updates = 0
        self.sample_count = 0
        self.episode_reward = 0.0
        self.best_score = -1e18
        self.last_score = 0.0
        self.feature_rows: list[np.ndarray] = []
        self.action_rows: list[np.ndarray] = []
        self.current_model = self._new_zero_model()
        self.best_model_state = self.current_model.policy

    def _new_zero_model(self) -> LinearPolicyModel:
        policy = LinearPolicy(
            weights=np.zeros((self.feature_dim, 2), dtype=np.float32),
            bias=np.array([0.2, 0.0], dtype=np.float32),
            feature_mean=np.zeros(self.feature_dim, dtype=np.float32),
            feature_std=np.ones(self.feature_dim, dtype=np.float32),
        )
        return LinearPolicyModel(policy=policy)

    def seed_from_linear_policy(self, policy: LinearPolicy) -> bool:
        if policy.weights.shape != (self.feature_dim, 2):
            return False
        seeded = LinearPolicy(
            weights=policy.weights.copy(),
            bias=policy.bias.copy(),
            feature_mean=policy.feature_mean.copy(),
            feature_std=policy.feature_std.copy(),
        )
        self.current_model = LinearPolicyModel(policy=seeded)
        self.best_model_state = seeded
        return True

    def _teacher_blend(self) -> float:
        progress = min(1.0, self.steps / float(max(1, self.config.teacher_blend_decay_steps)))
        start = self.config.teacher_blend_start
        end = self.config.teacher_blend_end
        return float(start + (end - start) * progress)

    def act(self, observation: dict, teacher_action: Action | None = None) -> Action:
        model_action = self.current_model.act(observation)
        if teacher_action is None:
            return model_action
        alpha = self._teacher_blend()
        throttle = alpha * teacher_action.throttle + (1.0 - alpha) * model_action.throttle
        steering = alpha * teacher_action.steering + (1.0 - alpha) * model_action.steering
        return Action(
            throttle=float(np.clip(throttle, -1.0, 1.0)),
            steering=float(np.clip(steering, -1.0, 1.0)),
        )

    def observe_teacher(self, observation: dict, teacher_action: Action) -> None:
        feature_vec = features_from_observation(observation).astype(np.float32)
        action_vec = np.array([teacher_action.throttle, teacher_action.steering], dtype=np.float32)
        self.feature_rows.append(feature_vec)
        self.action_rows.append(action_vec)
        self.sample_count += 1
        if len(self.feature_rows) > self.config.max_buffer:
            cut = len(self.feature_rows) - self.config.max_buffer
            self.feature_rows = self.feature_rows[cut:]
            self.action_rows = self.action_rows[cut:]

        self.steps += 1
        if self.steps % self.config.fit_every_steps != 0:
            return
        if len(self.feature_rows) < self.config.min_fit_samples:
            return

        x = np.stack(self.feature_rows, axis=0)
        y = np.stack(self.action_rows, axis=0)
        policy = fit_linear_policy(x, y, l2_reg=2e-3)
        self.current_model = LinearPolicyModel(policy=policy)
        self.fit_updates += 1

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
            self.best_model_state = LinearPolicy(
                weights=self.current_model.policy.weights.copy(),
                bias=self.current_model.policy.bias.copy(),
                feature_mean=self.current_model.policy.feature_mean.copy(),
                feature_std=self.current_model.policy.feature_std.copy(),
            )
        self.episode_reward = 0.0
        return True

    def status_label(self) -> str:
        blend = self._teacher_blend()
        return (
            "train-live "
            f"| fit={self.fit_updates} samples={self.sample_count} blend={blend:4.2f} "
            f"| last={self.last_score:6.1f} best={self.best_score:6.1f}"
        )

    def best_model(self) -> LinearPolicyModel:
        return LinearPolicyModel(policy=self.best_model_state)

    @staticmethod
    def feature_dim_from_observation(observation: dict) -> int:
        return len(features_from_observation(observation))

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np

from drivesim.core.types import Action
from drivesim.ml.models import PolicyModel, load_policy_model


class AssistAgent:
    """Placeholder agent for a future ML policy.

    The current behavior is a tiny heuristic that keeps the assistant mode
    usable before a learned model is integrated.
    """

    def __init__(self, model_path: str = "models/assist_policy.npz"):
        self.model_path = model_path
        self.policy: PolicyModel | None = None
        self.model_name = "none"
        if Path(model_path).exists():
            self.policy = load_policy_model(model_path)
            self.model_name = self.policy.model_name

    @staticmethod
    def default_model_path() -> str:
        return os.getenv("DRIVESIM_ASSIST_MODEL", "models/assist_policy.npz")

    def writable_model_path(self) -> str:
        if Path(self.model_path).suffix == ".npz":
            return self.model_path
        return "models/assist_policy.npz"

    @staticmethod
    def _goal_seek_action(observation: dict) -> Action:
        pose = np.asarray(observation.get("pose", [0.0, 0.0, 0.0, 0.0]), dtype=np.float32)
        goal = np.asarray(observation.get("goal", [0.0, 0.0]), dtype=np.float32)
        yaw = float(pose[2]) if pose.size >= 3 else 0.0
        dx = float(goal[0] - pose[0])
        dy = float(goal[1] - pose[1])
        heading = math.atan2(dy, dx)
        err = ((heading - yaw + math.pi) % (2.0 * math.pi)) - math.pi
        steering = float(np.clip(1.15 * err, -1.0, 1.0))

        front = float(min(observation.get("lidar_front", [99.0])))
        if front > 35.0:
            throttle = 0.75
        elif front > 24.0:
            throttle = 0.45
        else:
            throttle = 0.10
        if abs(err) > 1.3:
            throttle = min(throttle, 0.25)
        return Action(throttle=throttle, steering=steering)

    @staticmethod
    def _safety_override(observation: dict, proposed: Action) -> Action:
        lidar = np.asarray(observation.get("lidar", []), dtype=np.float32)
        front = float(min(observation.get("lidar_front", [99.0])))
        if lidar.size == 0:
            return proposed

        n = lidar.size
        third = max(1, n // 3)
        left_clear = float(np.mean(lidar[:third]))
        right_clear = float(np.mean(lidar[-third:]))
        turn = -0.9 if left_clear > right_clear else 0.9

        if front < 12.0:
            return Action(throttle=-0.45, steering=turn)
        if front < 20.0:
            steer = 0.7 * turn + 0.3 * proposed.steering
            return Action(throttle=-0.10, steering=float(np.clip(steer, -1.0, 1.0)))
        if front < 28.0 and proposed.throttle > 0.4:
            return Action(throttle=0.25, steering=proposed.steering)
        return proposed

    @property
    def mode_label(self) -> str:
        if self.policy is not None:
            return f"assistant (trained:{self.model_name})"
        return "assistant (heuristic)"

    def act(self, observation: dict) -> Action:
        if self.policy is not None:
            try:
                learned = self.policy.act(observation)
            except ValueError:
                return self._safety_override(observation, self._goal_seek_action(observation))
            prior = self._goal_seek_action(observation)
            front = float(min(observation.get("lidar_front", [99.0])))
            steer_disagreement = abs(learned.steering - prior.steering)
            throttle_disagreement = abs(learned.throttle - prior.throttle)

            # Reduce learned contribution when behavior disagrees sharply with the safe prior.
            learned_weight = 0.55
            if steer_disagreement > 0.7 or throttle_disagreement > 0.65:
                learned_weight = 0.25
            if front < 24.0:
                learned_weight = min(learned_weight, 0.20)
            prior_weight = 1.0 - learned_weight

            blended = Action(
                throttle=float(np.clip(learned_weight * learned.throttle + prior_weight * prior.throttle, -1.0, 1.0)),
                steering=float(np.clip(learned_weight * learned.steering + prior_weight * prior.steering, -1.0, 1.0)),
            )
            return self._safety_override(observation, blended)
        return self._safety_override(observation, self._goal_seek_action(observation))

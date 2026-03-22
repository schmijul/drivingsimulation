from __future__ import annotations

from pathlib import Path

from drivesim.core.types import Action
from drivesim.ml.policy import LinearPolicy


class AssistAgent:
    """Placeholder agent for a future ML policy.

    The current behavior is a tiny heuristic that keeps the assistant mode
    usable before a learned model is integrated.
    """

    def __init__(self, model_path: str = "models/assist_policy.npz"):
        self.model_path = model_path
        self.policy: LinearPolicy | None = None
        if Path(model_path).exists():
            self.policy = LinearPolicy.load(model_path)

    @property
    def mode_label(self) -> str:
        if self.policy is not None:
            return "assistant (trained)"
        return "assistant (heuristic)"

    def act(self, observation: dict) -> Action:
        if self.policy is not None:
            return self.policy.act(observation)
        front = min(observation.get("lidar_front", [99.0]))
        if front < 26.0:
            return Action(throttle=0.15, steering=0.85)
        return Action(throttle=0.7, steering=0.0)

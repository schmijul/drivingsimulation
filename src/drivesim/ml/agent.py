from __future__ import annotations

from drivesim.core.types import Action


class AssistAgent:
    """Placeholder agent for a future ML policy.

    The current behavior is a tiny heuristic that keeps the assistant mode
    usable before a learned model is integrated.
    """

    def act(self, observation: dict) -> Action:
        front = min(observation.get("lidar_front", [99.0]))
        if front < 26.0:
            return Action(throttle=0.15, steering=0.85)
        return Action(throttle=0.7, steering=0.0)

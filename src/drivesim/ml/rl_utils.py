from __future__ import annotations

from typing import Any

try:
    from stable_baselines3 import PPO
except ImportError:  # pragma: no cover - optional dependency
    PPO = None


def list_rl_algorithms() -> list[str]:
    return ["ppo"]


def require_sb3() -> None:
    if PPO is None:
        raise ImportError(
            "RL training requires 'stable-baselines3' and its dependencies. "
            "Install with: pip install -e .[rl]"
        )


def load_rl_model(path: str, algorithm: str = "ppo") -> Any:
    require_sb3()
    if algorithm != "ppo":
        raise ValueError(f"unsupported RL algorithm: {algorithm!r}")
    assert PPO is not None
    return PPO.load(path)

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from drivesim.core.types import Action
from drivesim.ml.policy import LinearPolicy, features_from_observation


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
        x = features_from_observation(observation).astype(np.float32)
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


def list_model_architectures() -> list[str]:
    return ["linear", "tiny_mlp"]


def load_policy_model(path: str) -> PolicyModel:
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

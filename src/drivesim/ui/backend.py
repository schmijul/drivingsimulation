from __future__ import annotations

from typing import Protocol

import numpy as np
import pygame

from drivesim.autonomy.sensors import LidarObservation
from drivesim.core.types import SimState


class RenderBackend(Protocol):
    """Renderer contract for future 2D/3D backends."""

    def frame_size(self) -> tuple[int, int]:
        ...

    def set_driving_view(self, view: str) -> None:
        ...

    def render(
        self,
        screen: pygame.Surface,
        state: SimState,
        grid: np.ndarray,
        grid_resolution: float,
        lidar: LidarObservation,
        mode: str,
    ) -> None:
        ...

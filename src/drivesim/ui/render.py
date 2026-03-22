from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import pygame

from drivesim.autonomy.sensors import LidarObservation
from drivesim.core.types import SimState
from drivesim.ui.backend import RenderBackend

Color = Tuple[int, int, int]

BG: Color = (16, 19, 22)
ROAD: Color = (31, 38, 44)
OBSTACLE: Color = (82, 72, 58)
GRID_FREE: Color = (38, 88, 68)
GRID_OCC: Color = (190, 76, 66)
CAR: Color = (255, 199, 102)
GOAL: Color = (84, 212, 146)
PATH: Color = (104, 163, 255)
RAY: Color = (96, 142, 181)
TEXT: Color = (238, 241, 245)


class Renderer2D(RenderBackend):
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        pygame.font.init()
        self.font = pygame.font.SysFont("dejavusansmono", 16)

    def _draw_world(self, surf: pygame.Surface, state: SimState) -> None:
        surf.fill(BG)
        pygame.draw.rect(surf, ROAD, pygame.Rect(0, 0, state.world.width, state.world.height), border_radius=18)
        for obs in state.world.obstacles:
            pygame.draw.rect(surf, OBSTACLE, pygame.Rect(obs.x, obs.y, obs.w, obs.h), border_radius=8)

    def _draw_grid(self, surf: pygame.Surface, grid: np.ndarray, resolution: float) -> None:
        rows, cols = grid.shape
        for gy in range(rows):
            for gx in range(cols):
                p = float(grid[gy, gx])
                if p < 0.08:
                    continue
                if p > 0.5:
                    color = GRID_OCC
                    alpha = min(155, int(40 + p * 130))
                else:
                    color = GRID_FREE
                    alpha = min(95, int(20 + p * 120))
                cell = pygame.Surface((resolution, resolution), pygame.SRCALPHA)
                cell.fill((*color, alpha))
                surf.blit(cell, (gx * resolution, gy * resolution))

    def _draw_lidar(self, surf: pygame.Surface, state: SimState, lidar: LidarObservation) -> None:
        x, y = state.vehicle.x, state.vehicle.y
        for a, d in zip(lidar.angles, lidar.distances):
            ex = x + math.cos(a) * d
            ey = y + math.sin(a) * d
            pygame.draw.line(surf, RAY, (x, y), (ex, ey), 1)

    def _draw_path(self, surf: pygame.Surface, path: list[tuple[float, float]]) -> None:
        if len(path) < 2:
            return
        pygame.draw.lines(surf, PATH, False, path, 3)

    def _draw_car(self, surf: pygame.Surface, state: SimState) -> None:
        x, y, yaw = state.vehicle.x, state.vehicle.y, state.vehicle.yaw
        nose = (x + math.cos(yaw) * 14, y + math.sin(yaw) * 14)
        left = (x + math.cos(yaw + 2.4) * 10, y + math.sin(yaw + 2.4) * 10)
        right = (x + math.cos(yaw - 2.4) * 10, y + math.sin(yaw - 2.4) * 10)
        pygame.draw.polygon(surf, CAR, [nose, left, right])

    def _draw_goal(self, surf: pygame.Surface, state: SimState) -> None:
        gx, gy = state.world.goal
        pygame.draw.circle(surf, GOAL, (int(gx), int(gy)), 11)
        pygame.draw.circle(surf, (10, 20, 14), (int(gx), int(gy)), 5)

    def _draw_hud(self, surf: pygame.Surface, mode: str, t: float, collided: bool) -> None:
        panel = pygame.Surface((330, 72), pygame.SRCALPHA)
        panel.fill((12, 14, 16, 170))
        surf.blit(panel, (14, 14))
        msg = f"mode={mode}   t={t:6.2f}s   {'COLLISION' if collided else 'RUNNING'}"
        txt = self.font.render(msg, True, TEXT)
        surf.blit(txt, (24, 38))

    def render(
        self,
        screen: pygame.Surface,
        state: SimState,
        grid: np.ndarray,
        grid_resolution: float,
        lidar: LidarObservation,
        mode: str,
    ) -> None:
        self._draw_world(screen, state)
        self._draw_grid(screen, grid, grid_resolution)
        self._draw_goal(screen, state)
        self._draw_path(screen, state.path)
        self._draw_lidar(screen, state, lidar)
        self._draw_car(screen, state)
        self._draw_hud(screen, mode, state.t, state.collided)

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
PANEL_BG: Color = (10, 12, 15)
PANEL_EDGE: Color = (58, 68, 78)
MAP_BG: Color = (22, 26, 31)


class Renderer2D(RenderBackend):
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.margin = 18
        self.header = 34
        pygame.font.init()
        self.font = pygame.font.SysFont("dejavusansmono", 16)
        self.title_font = pygame.font.SysFont("dejavusansmono", 18, bold=True)

    def frame_size(self) -> tuple[int, int]:
        total_width = self.width * 2 + self.margin * 3
        total_height = self.height + self.margin * 2 + self.header
        return total_width, total_height

    def _panel_rects(self) -> tuple[pygame.Rect, pygame.Rect]:
        top = self.margin + self.header
        left = pygame.Rect(self.margin, top, self.width, self.height)
        right = pygame.Rect(self.margin * 2 + self.width, top, self.width, self.height)
        return left, right

    def _draw_frame(self, screen: pygame.Surface) -> tuple[pygame.Rect, pygame.Rect]:
        screen.fill(BG)
        left_rect, right_rect = self._panel_rects()
        for rect in (left_rect, right_rect):
            pygame.draw.rect(screen, PANEL_BG, rect.inflate(8, 8), border_radius=18)
            pygame.draw.rect(screen, PANEL_EDGE, rect.inflate(8, 8), width=2, border_radius=18)

        left_title = self.title_font.render("Driving View", True, TEXT)
        right_title = self.title_font.render("Live SLAM Map", True, TEXT)
        screen.blit(left_title, (left_rect.x + 8, self.margin + 4))
        screen.blit(right_title, (right_rect.x + 8, self.margin + 4))
        return left_rect, right_rect

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

    def _draw_map_car(self, surf: pygame.Surface, state: SimState) -> None:
        x, y = int(state.vehicle.x), int(state.vehicle.y)
        pygame.draw.circle(surf, CAR, (x, y), 7)
        heading = (
            int(x + math.cos(state.vehicle.yaw) * 12),
            int(y + math.sin(state.vehicle.yaw) * 12),
        )
        pygame.draw.line(surf, BG, (x, y), heading, 2)

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

    def _draw_slam_view(self, surf: pygame.Surface, state: SimState, grid: np.ndarray, resolution: float) -> None:
        surf.fill(MAP_BG)
        self._draw_grid(surf, grid, resolution)
        self._draw_goal(surf, state)
        self._draw_path(surf, state.path)
        self._draw_map_car(surf, state)

    def render(
        self,
        screen: pygame.Surface,
        state: SimState,
        grid: np.ndarray,
        grid_resolution: float,
        lidar: LidarObservation,
        mode: str,
    ) -> None:
        left_rect, right_rect = self._draw_frame(screen)
        world_surface = pygame.Surface((self.width, self.height))
        map_surface = pygame.Surface((self.width, self.height))

        self._draw_world(world_surface, state)
        self._draw_goal(world_surface, state)
        self._draw_path(world_surface, state.path)
        self._draw_lidar(world_surface, state, lidar)
        self._draw_car(world_surface, state)
        self._draw_hud(world_surface, mode, state.t, state.collided)

        self._draw_slam_view(map_surface, state, grid, grid_resolution)
        self._draw_hud(map_surface, mode, state.t, state.collided)

        screen.blit(world_surface, left_rect.topleft)
        screen.blit(map_surface, right_rect.topleft)

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
MUTED: Color = (173, 186, 199)
OBSTACLE_SIDE: Color = (60, 52, 42)
OBSTACLE_TOP: Color = (118, 104, 88)
GROUND_LINE: Color = (46, 52, 60)


class Renderer2D(RenderBackend):
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.margin = 18
        self.header = 34
        self.driving_view = "isometric"
        self.iso_scale = 0.72
        self.iso_lift = 0.38
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

        left_title = self.title_font.render(f"Driving View ({self.driving_view})", True, TEXT)
        right_title = self.title_font.render("Live SLAM Map", True, TEXT)
        screen.blit(left_title, (left_rect.x + 8, self.margin + 4))
        screen.blit(right_title, (right_rect.x + 8, self.margin + 4))
        return left_rect, right_rect

    def set_driving_view(self, view: str) -> None:
        if view in ("topdown", "isometric"):
            self.driving_view = view

    def _draw_world(self, surf: pygame.Surface, state: SimState) -> None:
        surf.fill(BG)
        pygame.draw.rect(surf, ROAD, pygame.Rect(0, 0, state.world.width, state.world.height), border_radius=18)
        for obs in state.world.obstacles:
            pygame.draw.rect(surf, OBSTACLE, pygame.Rect(obs.x, obs.y, obs.w, obs.h), border_radius=8)

    def _iso_point(self, x: float, y: float, z: float, world_h: float) -> tuple[float, float]:
        sx = (x - y) * self.iso_scale + self.width * 0.5
        sy = (x + y) * self.iso_scale * 0.5 - z + self.height * 0.5 - world_h * self.iso_lift
        return sx, sy

    def _draw_world_iso(self, surf: pygame.Surface, state: SimState) -> None:
        surf.fill(BG)
        corners = [
            self._iso_point(0, 0, 0, state.world.height),
            self._iso_point(state.world.width, 0, 0, state.world.height),
            self._iso_point(state.world.width, state.world.height, 0, state.world.height),
            self._iso_point(0, state.world.height, 0, state.world.height),
        ]
        pygame.draw.polygon(surf, ROAD, corners)
        pygame.draw.lines(surf, GROUND_LINE, True, corners, 2)

        # Draw back-to-front by y for stable depth ordering.
        obstacles = sorted(state.world.obstacles, key=lambda o: o.y + o.h)
        for obs in obstacles:
            h = 28.0
            p1 = self._iso_point(obs.x, obs.y, 0, state.world.height)
            p2 = self._iso_point(obs.x + obs.w, obs.y, 0, state.world.height)
            p3 = self._iso_point(obs.x + obs.w, obs.y + obs.h, 0, state.world.height)
            p4 = self._iso_point(obs.x, obs.y + obs.h, 0, state.world.height)
            p1t = self._iso_point(obs.x, obs.y, h, state.world.height)
            p2t = self._iso_point(obs.x + obs.w, obs.y, h, state.world.height)
            p3t = self._iso_point(obs.x + obs.w, obs.y + obs.h, h, state.world.height)
            p4t = self._iso_point(obs.x, obs.y + obs.h, h, state.world.height)
            pygame.draw.polygon(surf, OBSTACLE_SIDE, [p1, p2, p2t, p1t])
            pygame.draw.polygon(surf, OBSTACLE_SIDE, [p2, p3, p3t, p2t])
            pygame.draw.polygon(surf, OBSTACLE_SIDE, [p4, p3, p3t, p4t])
            pygame.draw.polygon(surf, OBSTACLE_TOP, [p1t, p2t, p3t, p4t])

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

    def _draw_lidar_iso(self, surf: pygame.Surface, state: SimState, lidar: LidarObservation) -> None:
        x, y = state.vehicle.x, state.vehicle.y
        sx, sy = self._iso_point(x, y, 8.0, state.world.height)
        for a, d in zip(lidar.angles, lidar.distances):
            ex = x + math.cos(a) * d
            ey = y + math.sin(a) * d
            esx, esy = self._iso_point(ex, ey, 0.0, state.world.height)
            pygame.draw.line(surf, RAY, (sx, sy), (esx, esy), 1)

    def _draw_path(self, surf: pygame.Surface, path: list[tuple[float, float]]) -> None:
        if len(path) < 2:
            return
        pygame.draw.lines(surf, PATH, False, path, 3)

    def _draw_path_iso(self, surf: pygame.Surface, state: SimState, path: list[tuple[float, float]]) -> None:
        if len(path) < 2:
            return
        projected = [self._iso_point(x, y, 2.0, state.world.height) for x, y in path]
        pygame.draw.lines(surf, PATH, False, projected, 3)

    def _draw_car(self, surf: pygame.Surface, state: SimState) -> None:
        x, y, yaw = state.vehicle.x, state.vehicle.y, state.vehicle.yaw
        nose = (x + math.cos(yaw) * 14, y + math.sin(yaw) * 14)
        left = (x + math.cos(yaw + 2.4) * 10, y + math.sin(yaw + 2.4) * 10)
        right = (x + math.cos(yaw - 2.4) * 10, y + math.sin(yaw - 2.4) * 10)
        pygame.draw.polygon(surf, CAR, [nose, left, right])

    def _draw_car_iso(self, surf: pygame.Surface, state: SimState) -> None:
        x, y, yaw = state.vehicle.x, state.vehicle.y, state.vehicle.yaw
        nose_w = (x + math.cos(yaw) * 12, y + math.sin(yaw) * 12)
        left_w = (x + math.cos(yaw + 2.4) * 8, y + math.sin(yaw + 2.4) * 8)
        right_w = (x + math.cos(yaw - 2.4) * 8, y + math.sin(yaw - 2.4) * 8)
        nose = self._iso_point(nose_w[0], nose_w[1], 11.0, state.world.height)
        left = self._iso_point(left_w[0], left_w[1], 11.0, state.world.height)
        right = self._iso_point(right_w[0], right_w[1], 11.0, state.world.height)
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

    def _draw_goal_iso(self, surf: pygame.Surface, state: SimState) -> None:
        gx, gy = state.world.goal
        sx, sy = self._iso_point(gx, gy, 6.0, state.world.height)
        pygame.draw.circle(surf, GOAL, (int(sx), int(sy)), 9)
        pygame.draw.circle(surf, (10, 20, 14), (int(sx), int(sy)), 4)

    def _draw_hud(self, surf: pygame.Surface, mode: str, t: float, collided: bool) -> None:
        panel = pygame.Surface((330, 72), pygame.SRCALPHA)
        panel.fill((12, 14, 16, 170))
        surf.blit(panel, (14, 14))
        msg = f"mode={mode}   t={t:6.2f}s   {'COLLISION' if collided else 'RUNNING'}"
        txt = self.font.render(msg, True, TEXT)
        surf.blit(txt, (24, 38))

    def _draw_controls(self, surf: pygame.Surface) -> None:
        panel = pygame.Surface((370, 118), pygame.SRCALPHA)
        panel.fill((9, 11, 14, 178))
        x = 12
        y = self.height - 130
        surf.blit(panel, (x, y))
        lines = [
            "W/S throttle-brake  A/D steer",
            "TAB switch mode     R reset",
            "V view topdown/iso  C clear replay",
            "ESC quit",
        ]
        title = self.font.render("Controls", True, TEXT)
        surf.blit(title, (x + 10, y + 10))
        for i, line in enumerate(lines):
            txt = self.font.render(line, True, MUTED)
            surf.blit(txt, (x + 10, y + 36 + i * 24))

    def _draw_slam_legend(self, surf: pygame.Surface, grid: np.ndarray) -> None:
        panel = pygame.Surface((260, 120), pygame.SRCALPHA)
        panel.fill((9, 11, 14, 178))
        x = self.width - 272
        y = self.height - 132
        surf.blit(panel, (x, y))
        title = self.font.render("SLAM Legend", True, TEXT)
        surf.blit(title, (x + 10, y + 10))

        explored = float(np.count_nonzero(grid > 0.08)) / float(grid.size)
        stats = self.font.render(f"explored: {explored * 100.0:4.1f}%", True, MUTED)
        surf.blit(stats, (x + 10, y + 34))

        pygame.draw.rect(surf, GRID_FREE, pygame.Rect(x + 10, y + 62, 14, 14))
        surf.blit(self.font.render("free evidence", True, MUTED), (x + 30, y + 60))
        pygame.draw.rect(surf, GRID_OCC, pygame.Rect(x + 10, y + 86, 14, 14))
        surf.blit(self.font.render("occupied evidence", True, MUTED), (x + 30, y + 84))

    def _draw_slam_view(self, surf: pygame.Surface, state: SimState, grid: np.ndarray, resolution: float) -> None:
        surf.fill(MAP_BG)
        self._draw_grid(surf, grid, resolution)
        self._draw_goal(surf, state)
        self._draw_path(surf, state.path)
        self._draw_map_car(surf, state)
        self._draw_slam_legend(surf, grid)

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

        if self.driving_view == "isometric":
            self._draw_world_iso(world_surface, state)
            self._draw_goal_iso(world_surface, state)
            self._draw_path_iso(world_surface, state, state.path)
            self._draw_lidar_iso(world_surface, state, lidar)
            self._draw_car_iso(world_surface, state)
        else:
            self._draw_world(world_surface, state)
            self._draw_goal(world_surface, state)
            self._draw_path(world_surface, state.path)
            self._draw_lidar(world_surface, state, lidar)
            self._draw_car(world_surface, state)
        self._draw_hud(world_surface, mode, state.t, state.collided)
        self._draw_controls(world_surface)

        self._draw_slam_view(map_surface, state, grid, grid_resolution)
        self._draw_hud(map_surface, mode, state.t, state.collided)

        screen.blit(world_surface, left_rect.topleft)
        screen.blit(map_surface, right_rect.topleft)

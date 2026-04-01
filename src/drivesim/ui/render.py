from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import pygame

from drivesim.autonomy.sensors import LidarObservation
from drivesim.core.types import SimState
from drivesim.ui.backend import RenderBackend

Color = Tuple[int, int, int]

# --- Refined dark palette ---------------------------------------------------
BG: Color = (13, 15, 18)
ROAD: Color = (28, 34, 40)
OBSTACLE: Color = (82, 72, 58)
GRID_FREE: Color = (34, 197, 128)
GRID_OCC: Color = (239, 68, 68)
CAR: Color = (250, 196, 80)
CAR_OUTLINE: Color = (210, 160, 50)
GOAL: Color = (52, 211, 153)
GOAL_INNER: Color = (16, 185, 129)
PATH: Color = (96, 165, 250)
RAY: Color = (71, 115, 162)
TEXT: Color = (241, 245, 249)
TEXT_DIM: Color = (148, 163, 184)
PANEL_BG: Color = (15, 18, 22)
PANEL_EDGE: Color = (38, 45, 55)
PANEL_HEADER: Color = (20, 24, 30)
MAP_BG: Color = (18, 22, 28)
MUTED: Color = (148, 163, 184)
OBSTACLE_SIDE: Color = (60, 52, 42)
OBSTACLE_TOP: Color = (118, 104, 88)
GROUND_LINE: Color = (38, 45, 55)
OBSTACLE_CHASE_SIDE: Color = (70, 60, 48)
OBSTACLE_CHASE_TOP: Color = (128, 112, 92)
DYNAMIC_OBSTACLE: Color = (251, 146, 60)
DYNAMIC_OBSTACLE_SIDE: Color = (194, 110, 46)
DYNAMIC_OBSTACLE_TOP: Color = (253, 186, 116)
ACCENT_BLUE: Color = (59, 130, 246)
ACCENT_GREEN: Color = (34, 197, 94)
ACCENT_RED: Color = (239, 68, 68)
ACCENT_AMBER: Color = (245, 158, 11)
KEY_BG: Color = (30, 36, 44)
KEY_BORDER: Color = (55, 65, 80)
SEPARATOR: Color = (30, 36, 44)
BADGE_MANUAL: Color = (59, 130, 246)
BADGE_AUTOPILOT: Color = (168, 85, 247)
BADGE_ASSISTANT: Color = (34, 197, 94)
BADGE_TRAIN: Color = (245, 158, 11)


def _lerp_color(a: Color, b: Color, t: float) -> Color:
    return (
        int(a[0] + (b[0] - a[0]) * t),
        int(a[1] + (b[1] - a[1]) * t),
        int(a[2] + (b[2] - a[2]) * t),
    )


class Renderer2D(RenderBackend):
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.margin = 16
        self.header = 44
        self.driving_view = "chase"
        self.camera_mode = "follow"
        self.iso_scale = 0.72
        self.iso_lift = 0.38
        self.cam_x = width * 0.5
        self.cam_y = height * 0.5
        pygame.font.init()
        self.font = pygame.font.SysFont("dejavusans", 13)
        self.font_mono = pygame.font.SysFont("dejavusansmono", 13)
        self.title_font = pygame.font.SysFont("dejavusans", 14, bold=True)
        self.hud_font = pygame.font.SysFont("dejavusans", 12)
        self.badge_font = pygame.font.SysFont("dejavusans", 11, bold=True)
        self.key_font = pygame.font.SysFont("dejavusansmono", 11, bold=True)

    def frame_size(self) -> tuple[int, int]:
        total_width = self.width * 2 + self.margin * 3
        total_height = self.height + self.margin * 2 + self.header
        return total_width, total_height

    def _panel_rects(self) -> tuple[pygame.Rect, pygame.Rect]:
        top = self.margin + self.header
        left = pygame.Rect(self.margin, top, self.width, self.height)
        right = pygame.Rect(self.margin * 2 + self.width, top, self.width, self.height)
        return left, right

    def _draw_rounded_panel(
        self, screen: pygame.Surface, rect: pygame.Rect, title: str, subtitle: str = ""
    ) -> None:
        outer = rect.inflate(6, 6)
        # Shadow
        shadow = pygame.Surface((outer.w + 8, outer.h + 8), pygame.SRCALPHA)
        pygame.draw.rect(shadow, (0, 0, 0, 40), shadow.get_rect(), border_radius=14)
        screen.blit(shadow, (outer.x - 2, outer.y + 2))
        # Panel body
        pygame.draw.rect(screen, PANEL_BG, outer, border_radius=12)
        pygame.draw.rect(screen, PANEL_EDGE, outer, width=1, border_radius=12)
        # Header bar
        header_h = 32
        header_rect = pygame.Rect(outer.x + 1, outer.y + 1, outer.w - 2, header_h)
        header_surf = pygame.Surface((header_rect.w, header_rect.h), pygame.SRCALPHA)
        pygame.draw.rect(
            header_surf,
            (*PANEL_HEADER, 255),
            header_surf.get_rect(),
            border_top_left_radius=11,
            border_top_right_radius=11,
        )
        screen.blit(header_surf, header_rect.topleft)
        # Separator line under header
        pygame.draw.line(
            screen,
            SEPARATOR,
            (outer.x + 1, outer.y + header_h + 1),
            (outer.x + outer.w - 2, outer.y + header_h + 1),
        )
        # Title text
        title_surf = self.title_font.render(title, True, TEXT)
        screen.blit(title_surf, (outer.x + 14, outer.y + 8))
        if subtitle:
            sub_surf = self.hud_font.render(subtitle, True, TEXT_DIM)
            screen.blit(sub_surf, (outer.x + 14 + title_surf.get_width() + 10, outer.y + 10))

    def _draw_frame(self, screen: pygame.Surface) -> tuple[pygame.Rect, pygame.Rect]:
        screen.fill(BG)
        left_rect, right_rect = self._panel_rects()

        view_label = "3D" if self.driving_view == "isometric" else self.driving_view.capitalize()
        cam_label = self.camera_mode.capitalize()
        self._draw_rounded_panel(
            screen, left_rect, "Driving View", f"{view_label} / {cam_label}"
        )
        self._draw_rounded_panel(screen, right_rect, "Occupancy Map")

        return left_rect, right_rect

    def set_driving_view(self, view: str) -> None:
        if view == "3d":
            view = "isometric"
        if view in ("topdown", "isometric", "chase"):
            self.driving_view = view

    def set_camera_mode(self, mode: str) -> None:
        if mode in ("tactical", "follow", "cinematic"):
            self.camera_mode = mode

    def _update_camera(self, state: SimState) -> None:
        target_x = state.world.width * 0.5
        target_y = state.world.height * 0.5
        smoothing = 0.08
        lookahead = 0.0

        if self.camera_mode == "follow":
            lookahead = 35.0
            smoothing = 0.14
            target_x = state.vehicle.x + math.cos(state.vehicle.yaw) * lookahead
            target_y = state.vehicle.y + math.sin(state.vehicle.yaw) * lookahead
        elif self.camera_mode == "cinematic":
            speed_boost = min(26.0, abs(state.vehicle.speed) * 0.35)
            lookahead = 50.0 + speed_boost
            smoothing = 0.09
            target_x = state.vehicle.x + math.cos(state.vehicle.yaw) * lookahead
            target_y = state.vehicle.y + math.sin(state.vehicle.yaw) * lookahead

        self.cam_x += (target_x - self.cam_x) * smoothing
        self.cam_y += (target_y - self.cam_y) * smoothing

    def _topdown_point(self, x: float, y: float) -> tuple[float, float]:
        return x - self.cam_x + self.width * 0.5, y - self.cam_y + self.height * 0.5

    def _draw_world(self, surf: pygame.Surface, state: SimState) -> None:
        surf.fill(BG)
        w = state.world.width
        h = state.world.height
        c1 = self._topdown_point(0, 0)
        c2 = self._topdown_point(w, 0)
        c3 = self._topdown_point(w, h)
        c4 = self._topdown_point(0, h)
        pygame.draw.polygon(surf, ROAD, [c1, c2, c3, c4])
        pygame.draw.lines(surf, GROUND_LINE, True, [c1, c2, c3, c4], 2)
        for obs in state.world.obstacles:
            ox, oy = self._topdown_point(obs.x, obs.y)
            pygame.draw.rect(surf, OBSTACLE, pygame.Rect(ox, oy, obs.w, obs.h), border_radius=8)
        for obs in state.world.dynamic_obstacles:
            ox, oy = self._topdown_point(obs.x, obs.y)
            pygame.draw.rect(surf, DYNAMIC_OBSTACLE, pygame.Rect(ox, oy, obs.w, obs.h), border_radius=7)

    def _iso_point(self, x: float, y: float, z: float, world_h: float) -> tuple[float, float]:
        rel_x = x - self.cam_x
        rel_y = y - self.cam_y
        scale = self.iso_scale
        if self.camera_mode == "cinematic":
            speed_zoom = min(0.13, abs(rel_x) * 0.00008 + abs(rel_y) * 0.00008)
            scale *= 1.0 + speed_zoom
        sx = (rel_x - rel_y) * scale + self.width * 0.5
        sy = (rel_x + rel_y) * scale * 0.5 - z + self.height * 0.5 - world_h * self.iso_lift
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

        dynamic_obstacles = sorted(state.world.dynamic_obstacles, key=lambda o: o.y + o.h)
        for obs in dynamic_obstacles:
            h = 20.0
            p1 = self._iso_point(obs.x, obs.y, 0, state.world.height)
            p2 = self._iso_point(obs.x + obs.w, obs.y, 0, state.world.height)
            p3 = self._iso_point(obs.x + obs.w, obs.y + obs.h, 0, state.world.height)
            p4 = self._iso_point(obs.x, obs.y + obs.h, 0, state.world.height)
            p1t = self._iso_point(obs.x, obs.y, h, state.world.height)
            p2t = self._iso_point(obs.x + obs.w, obs.y, h, state.world.height)
            p3t = self._iso_point(obs.x + obs.w, obs.y + obs.h, h, state.world.height)
            p4t = self._iso_point(obs.x, obs.y + obs.h, h, state.world.height)
            pygame.draw.polygon(surf, DYNAMIC_OBSTACLE_SIDE, [p1, p2, p2t, p1t])
            pygame.draw.polygon(surf, DYNAMIC_OBSTACLE_SIDE, [p2, p3, p3t, p2t])
            pygame.draw.polygon(surf, DYNAMIC_OBSTACLE_SIDE, [p4, p3, p3t, p4t])
            pygame.draw.polygon(surf, DYNAMIC_OBSTACLE_TOP, [p1t, p2t, p3t, p4t])

    def _chase_point(self, wx: float, wy: float, state: SimState, z: float = 0.0) -> tuple[float, float]:
        vx = state.vehicle.x
        vy = state.vehicle.y
        yaw = state.vehicle.yaw
        dx = wx - vx
        dy = wy - vy

        # Convert world position into car-local coordinates.
        fwd = dx * math.cos(yaw) + dy * math.sin(yaw)
        lat = -dx * math.sin(yaw) + dy * math.cos(yaw)

        depth = max(-40.0, fwd)
        perspective = 1.0 / (1.0 + max(0.0, depth) * 0.006)
        scale = 1.35 * perspective
        sx = self.width * 0.5 + lat * scale
        sy = self.height * 0.84 - depth * scale - z * (0.45 + perspective * 0.75)
        return sx, sy

    def _draw_world_chase(self, surf: pygame.Surface, state: SimState) -> None:
        surf.fill(BG)
        world = state.world
        corners = [
            self._chase_point(0, 0, state),
            self._chase_point(world.width, 0, state),
            self._chase_point(world.width, world.height, state),
            self._chase_point(0, world.height, state),
        ]
        pygame.draw.polygon(surf, ROAD, corners)
        pygame.draw.lines(surf, GROUND_LINE, True, corners, 2)

        def _forward_depth(wx: float, wy: float) -> float:
            yaw = state.vehicle.yaw
            return (wx - state.vehicle.x) * math.cos(yaw) + (wy - state.vehicle.y) * math.sin(yaw)

        # Painter order: far objects first, near objects last.
        obstacles = sorted(
            state.world.obstacles,
            key=lambda o: _forward_depth(o.x + o.w * 0.5, o.y + o.h * 0.5),
            reverse=True,
        )
        for obs in obstacles:
            height = 24.0
            b1 = self._chase_point(obs.x, obs.y, state, 0.0)
            b2 = self._chase_point(obs.x + obs.w, obs.y, state, 0.0)
            b3 = self._chase_point(obs.x + obs.w, obs.y + obs.h, state, 0.0)
            b4 = self._chase_point(obs.x, obs.y + obs.h, state, 0.0)
            t1 = self._chase_point(obs.x, obs.y, state, height)
            t2 = self._chase_point(obs.x + obs.w, obs.y, state, height)
            t3 = self._chase_point(obs.x + obs.w, obs.y + obs.h, state, height)
            t4 = self._chase_point(obs.x, obs.y + obs.h, state, height)

            sides = [
                [b1, b2, t2, t1],
                [b2, b3, t3, t2],
                [b3, b4, t4, t3],
                [b4, b1, t1, t4],
            ]
            for poly in sides:
                pygame.draw.polygon(surf, OBSTACLE_CHASE_SIDE, poly)
            pygame.draw.polygon(surf, OBSTACLE_CHASE_TOP, [t1, t2, t3, t4])
            pygame.draw.lines(surf, OBSTACLE_SIDE, True, [b1, b2, b3, b4], 1)

        dynamic_obstacles = sorted(
            state.world.dynamic_obstacles,
            key=lambda o: _forward_depth(o.x + o.w * 0.5, o.y + o.h * 0.5),
            reverse=True,
        )
        for obs in dynamic_obstacles:
            height = 18.0
            b1 = self._chase_point(obs.x, obs.y, state, 0.0)
            b2 = self._chase_point(obs.x + obs.w, obs.y, state, 0.0)
            b3 = self._chase_point(obs.x + obs.w, obs.y + obs.h, state, 0.0)
            b4 = self._chase_point(obs.x, obs.y + obs.h, state, 0.0)
            t1 = self._chase_point(obs.x, obs.y, state, height)
            t2 = self._chase_point(obs.x + obs.w, obs.y, state, height)
            t3 = self._chase_point(obs.x + obs.w, obs.y + obs.h, state, height)
            t4 = self._chase_point(obs.x, obs.y + obs.h, state, height)

            sides = [
                [b1, b2, t2, t1],
                [b2, b3, t3, t2],
                [b3, b4, t4, t3],
                [b4, b1, t1, t4],
            ]
            for poly in sides:
                pygame.draw.polygon(surf, DYNAMIC_OBSTACLE_SIDE, poly)
            pygame.draw.polygon(surf, DYNAMIC_OBSTACLE_TOP, [t1, t2, t3, t4])
            pygame.draw.lines(surf, DYNAMIC_OBSTACLE_SIDE, True, [b1, b2, b3, b4], 1)

    def _draw_grid(self, surf: pygame.Surface, grid: np.ndarray, resolution: float) -> None:
        rows, cols = grid.shape
        res = int(resolution)
        for gy in range(rows):
            for gx in range(cols):
                p = float(grid[gy, gx])
                if p < 0.08:
                    continue
                if p > 0.5:
                    t = min(1.0, (p - 0.5) * 2.0)
                    color = _lerp_color((180, 50, 50), GRID_OCC, t)
                    alpha = min(180, int(60 + t * 120))
                else:
                    t = min(1.0, p / 0.5)
                    color = _lerp_color((25, 60, 45), GRID_FREE, t)
                    alpha = min(120, int(20 + t * 100))
                cell = pygame.Surface((res, res), pygame.SRCALPHA)
                cell.fill((*color, alpha))
                surf.blit(cell, (gx * res, gy * res))

    def _draw_lidar(self, surf: pygame.Surface, state: SimState, lidar: LidarObservation) -> None:
        x, y = state.vehicle.x, state.vehicle.y
        sx, sy = self._topdown_point(x, y)
        for a, d in zip(lidar.angles, lidar.distances):
            ex = x + math.cos(a) * d
            ey = y + math.sin(a) * d
            esx, esy = self._topdown_point(ex, ey)
            pygame.draw.line(surf, RAY, (sx, sy), (esx, esy), 1)

    def _draw_lidar_iso(self, surf: pygame.Surface, state: SimState, lidar: LidarObservation) -> None:
        x, y = state.vehicle.x, state.vehicle.y
        sx, sy = self._iso_point(x, y, 8.0, state.world.height)
        for a, d in zip(lidar.angles, lidar.distances):
            ex = x + math.cos(a) * d
            ey = y + math.sin(a) * d
            esx, esy = self._iso_point(ex, ey, 0.0, state.world.height)
            pygame.draw.line(surf, RAY, (sx, sy), (esx, esy), 1)

    def _draw_lidar_chase(self, surf: pygame.Surface, state: SimState, lidar: LidarObservation) -> None:
        sx, sy = self._chase_point(state.vehicle.x, state.vehicle.y, state)
        for a, d in zip(lidar.angles, lidar.distances):
            ex = state.vehicle.x + math.cos(a) * d
            ey = state.vehicle.y + math.sin(a) * d
            esx, esy = self._chase_point(ex, ey, state)
            pygame.draw.line(surf, RAY, (sx, sy), (esx, esy), 1)

    def _draw_path_with_glow(self, surf: pygame.Surface, points: list[tuple[float, float]]) -> None:
        if len(points) < 2:
            return
        glow = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
        pygame.draw.lines(glow, (*PATH, 35), False, points, 8)
        surf.blit(glow, (0, 0))
        pygame.draw.lines(surf, PATH, False, points, 2)

    def _draw_path(self, surf: pygame.Surface, path: list[tuple[float, float]]) -> None:
        projected = [self._topdown_point(x, y) for x, y in path]
        self._draw_path_with_glow(surf, projected)

    def _draw_path_iso(self, surf: pygame.Surface, state: SimState, path: list[tuple[float, float]]) -> None:
        projected = [self._iso_point(x, y, 2.0, state.world.height) for x, y in path]
        self._draw_path_with_glow(surf, projected)

    def _draw_path_chase(self, surf: pygame.Surface, state: SimState, path: list[tuple[float, float]]) -> None:
        projected = [self._chase_point(x, y, state) for x, y in path]
        self._draw_path_with_glow(surf, projected)

    def _draw_glow(self, surf: pygame.Surface, cx: float, cy: float, radius: int, color: Color, alpha: int = 40) -> None:
        glow = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow, (*color, alpha), (radius, radius), radius)
        surf.blit(glow, (int(cx) - radius, int(cy) - radius))

    def _draw_car(self, surf: pygame.Surface, state: SimState) -> None:
        x, y, yaw = state.vehicle.x, state.vehicle.y, state.vehicle.yaw
        sx, sy = self._topdown_point(x, y)
        self._draw_glow(surf, sx, sy, 20, CAR, 30)
        nose = self._topdown_point(x + math.cos(yaw) * 14, y + math.sin(yaw) * 14)
        left = self._topdown_point(x + math.cos(yaw + 2.4) * 10, y + math.sin(yaw + 2.4) * 10)
        right = self._topdown_point(x + math.cos(yaw - 2.4) * 10, y + math.sin(yaw - 2.4) * 10)
        pygame.draw.polygon(surf, CAR, [nose, left, right])
        pygame.draw.polygon(surf, CAR_OUTLINE, [nose, left, right], 2)

    def _draw_car_iso(self, surf: pygame.Surface, state: SimState) -> None:
        x, y, yaw = state.vehicle.x, state.vehicle.y, state.vehicle.yaw
        nose_w = (x + math.cos(yaw) * 12, y + math.sin(yaw) * 12)
        left_w = (x + math.cos(yaw + 2.4) * 8, y + math.sin(yaw + 2.4) * 8)
        right_w = (x + math.cos(yaw - 2.4) * 8, y + math.sin(yaw - 2.4) * 8)
        nose = self._iso_point(nose_w[0], nose_w[1], 11.0, state.world.height)
        left = self._iso_point(left_w[0], left_w[1], 11.0, state.world.height)
        right = self._iso_point(right_w[0], right_w[1], 11.0, state.world.height)
        cx = (nose[0] + left[0] + right[0]) / 3
        cy = (nose[1] + left[1] + right[1]) / 3
        self._draw_glow(surf, cx, cy, 18, CAR, 28)
        pygame.draw.polygon(surf, CAR, [nose, left, right])
        pygame.draw.polygon(surf, CAR_OUTLINE, [nose, left, right], 2)

    def _draw_car_chase(self, surf: pygame.Surface) -> None:
        cx = self.width * 0.5
        cy = self.height * 0.84
        self._draw_glow(surf, cx, cy, 22, CAR, 28)
        nose = (cx, cy - 16)
        left = (cx - 12, cy + 8)
        right = (cx + 12, cy + 8)
        pygame.draw.polygon(surf, CAR, [nose, left, right])
        pygame.draw.polygon(surf, CAR_OUTLINE, [nose, left, right], 2)

    def _draw_map_car(self, surf: pygame.Surface, state: SimState) -> None:
        x, y = int(state.vehicle.x), int(state.vehicle.y)
        self._draw_glow(surf, x, y, 14, CAR, 35)
        pygame.draw.circle(surf, CAR, (x, y), 6)
        pygame.draw.circle(surf, CAR_OUTLINE, (x, y), 6, 1)
        heading = (
            int(x + math.cos(state.vehicle.yaw) * 12),
            int(y + math.sin(state.vehicle.yaw) * 12),
        )
        pygame.draw.line(surf, CAR, (x, y), heading, 2)

    def _draw_goal(self, surf: pygame.Surface, state: SimState) -> None:
        gx, gy = state.world.goal
        sx, sy = self._topdown_point(gx, gy)
        self._draw_glow(surf, sx, sy, 22, GOAL, 35)
        pygame.draw.circle(surf, GOAL, (int(sx), int(sy)), 11)
        pygame.draw.circle(surf, GOAL_INNER, (int(sx), int(sy)), 6)
        pygame.draw.circle(surf, (220, 255, 230), (int(sx), int(sy)), 2)

    def _draw_goal_iso(self, surf: pygame.Surface, state: SimState) -> None:
        gx, gy = state.world.goal
        sx, sy = self._iso_point(gx, gy, 6.0, state.world.height)
        self._draw_glow(surf, sx, sy, 18, GOAL, 30)
        pygame.draw.circle(surf, GOAL, (int(sx), int(sy)), 9)
        pygame.draw.circle(surf, GOAL_INNER, (int(sx), int(sy)), 5)
        pygame.draw.circle(surf, (220, 255, 230), (int(sx), int(sy)), 2)

    def _draw_goal_chase(self, surf: pygame.Surface, state: SimState) -> None:
        gx, gy = state.world.goal
        sx, sy = self._chase_point(gx, gy, state)
        self._draw_glow(surf, sx, sy, 18, GOAL, 30)
        pygame.draw.circle(surf, GOAL, (int(sx), int(sy)), 9)
        pygame.draw.circle(surf, GOAL_INNER, (int(sx), int(sy)), 5)
        pygame.draw.circle(surf, (220, 255, 230), (int(sx), int(sy)), 2)

    def _draw_badge(self, surf: pygame.Surface, x: int, y: int, label: str, color: Color) -> int:
        txt = self.badge_font.render(label.upper(), True, TEXT)
        tw, th = txt.get_size()
        pw, ph = tw + 14, th + 6
        badge = pygame.Surface((pw, ph), pygame.SRCALPHA)
        pygame.draw.rect(badge, (*color, 200), badge.get_rect(), border_radius=4)
        badge.blit(txt, (7, 3))
        surf.blit(badge, (x, y))
        return pw + 6

    def _draw_key_badge(self, surf: pygame.Surface, x: int, y: int, key: str) -> int:
        txt = self.key_font.render(key, True, TEXT)
        tw, th = txt.get_size()
        pw, ph = tw + 10, th + 6
        badge = pygame.Surface((pw, ph), pygame.SRCALPHA)
        pygame.draw.rect(badge, (*KEY_BG, 220), badge.get_rect(), border_radius=3)
        pygame.draw.rect(badge, (*KEY_BORDER, 180), badge.get_rect(), width=1, border_radius=3)
        badge.blit(txt, (5, 3))
        surf.blit(badge, (x, y))
        return pw + 4

    def _mode_badge_color(self, mode: str) -> Color:
        if "autopilot" in mode:
            return BADGE_AUTOPILOT
        if "assistant" in mode or "policy" in mode.lower():
            return BADGE_ASSISTANT
        if "train" in mode:
            return BADGE_TRAIN
        return BADGE_MANUAL

    def _draw_hud(self, surf: pygame.Surface, mode: str, t: float, collided: bool) -> None:
        hud_h = 36
        panel = pygame.Surface((self.width, hud_h), pygame.SRCALPHA)
        pygame.draw.rect(panel, (10, 12, 16, 200), panel.get_rect())
        # Bottom edge highlight
        pygame.draw.line(
            panel, (*PANEL_EDGE, 120), (0, hud_h - 1), (self.width, hud_h - 1)
        )
        surf.blit(panel, (0, 0))

        cx = 12
        cy = 8

        # Mode badge - extract primary mode word for badge
        parts = mode.split("|")
        primary = parts[0].strip()
        badge_color = self._mode_badge_color(primary)
        cx += self._draw_badge(surf, cx, cy, primary, badge_color)

        # Extra info badges (map, expand, etc.)
        for part in parts[1:]:
            txt = self.hud_font.render(part.strip(), True, TEXT_DIM)
            surf.blit(txt, (cx + 2, cy + 4))
            cx += txt.get_width() + 12

        # Right side: time + status
        status_color = ACCENT_RED if collided else ACCENT_GREEN
        status_text = "COLLISION" if collided else "OK"
        time_txt = self.font_mono.render(f"{t:6.1f}s", True, TEXT_DIM)
        status_txt = self.badge_font.render(status_text, True, TEXT)
        st_w = status_txt.get_width() + 14
        st_x = self.width - st_w - 12
        # Status pill
        pill = pygame.Surface((st_w, 20), pygame.SRCALPHA)
        pygame.draw.rect(pill, (*status_color, 180), pill.get_rect(), border_radius=4)
        pill.blit(status_txt, (7, 3))
        surf.blit(pill, (st_x, cy))
        # Time
        surf.blit(time_txt, (st_x - time_txt.get_width() - 10, cy + 3))

    def _draw_controls(self, surf: pygame.Surface) -> None:
        entries = [
            ("W", "S", "Throttle / Brake"),
            ("A", "D", "Steer Left / Right"),
            ("TAB", "", "Cycle Mode"),
            ("R", "", "Reset Episode"),
            ("M", "", "Switch Map"),
            ("V", "", "Cycle View"),
            ("F", "", "Cycle Camera"),
            ("E", "", "Toggle Expand"),
            ("P", "", "Save Live Model"),
            ("C", "", "Clear Replay"),
            ("H", "", "Toggle Help"),
            ("ESC", "", "Quit"),
        ]
        row_h = 22
        pad = 14
        title_h = 30
        panel_h = title_h + len(entries) * row_h + pad
        panel_w = 260

        x = self.width - panel_w - 12
        y = 46

        panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        pygame.draw.rect(panel, (12, 14, 18, 215), panel.get_rect(), border_radius=10)
        pygame.draw.rect(panel, (*PANEL_EDGE, 140), panel.get_rect(), width=1, border_radius=10)
        surf.blit(panel, (x, y))

        title = self.title_font.render("Keyboard Shortcuts", True, TEXT)
        surf.blit(title, (x + pad, y + 8))
        pygame.draw.line(
            surf, (*SEPARATOR, 160),
            (x + pad, y + title_h - 2), (x + panel_w - pad, y + title_h - 2),
        )

        for i, (k1, k2, desc) in enumerate(entries):
            ry = y + title_h + i * row_h + 2
            kx = x + pad
            kx += self._draw_key_badge(surf, kx, ry, k1)
            if k2:
                kx += self._draw_key_badge(surf, kx, ry, k2)
            desc_txt = self.font.render(desc, True, MUTED)
            surf.blit(desc_txt, (kx + 4, ry + 2))

    def _draw_slam_legend(self, surf: pygame.Surface, grid: np.ndarray) -> None:
        panel_w, panel_h = 220, 100
        x = self.width - panel_w - 10
        y = self.height - panel_h - 10

        panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        pygame.draw.rect(panel, (12, 14, 18, 210), panel.get_rect(), border_radius=8)
        pygame.draw.rect(panel, (*PANEL_EDGE, 120), panel.get_rect(), width=1, border_radius=8)
        surf.blit(panel, (x, y))

        explored = float(np.count_nonzero(grid > 0.08)) / float(grid.size)

        # Progress bar
        bar_x, bar_y = x + 12, y + 12
        bar_w, bar_h = panel_w - 24, 6
        pygame.draw.rect(surf, (*KEY_BG, 255), pygame.Rect(bar_x, bar_y, bar_w, bar_h), border_radius=3)
        fill_w = max(2, int(bar_w * explored))
        pygame.draw.rect(surf, ACCENT_BLUE, pygame.Rect(bar_x, bar_y, fill_w, bar_h), border_radius=3)

        pct_txt = self.hud_font.render(f"Explored  {explored * 100.0:.1f}%", True, TEXT_DIM)
        surf.blit(pct_txt, (bar_x, bar_y + 12))

        # Legend items
        iy = bar_y + 32
        pygame.draw.rect(surf, GRID_FREE, pygame.Rect(bar_x, iy + 2, 10, 10), border_radius=2)
        surf.blit(self.hud_font.render("Free", True, MUTED), (bar_x + 16, iy))
        pygame.draw.rect(surf, GRID_OCC, pygame.Rect(bar_x + 80, iy + 2, 10, 10), border_radius=2)
        surf.blit(self.hud_font.render("Occupied", True, MUTED), (bar_x + 96, iy))

    def _draw_goal_map(self, surf: pygame.Surface, state: SimState) -> None:
        gx, gy = state.world.goal
        self._draw_glow(surf, gx, gy, 18, GOAL, 30)
        pygame.draw.circle(surf, GOAL, (int(gx), int(gy)), 10)
        pygame.draw.circle(surf, GOAL_INNER, (int(gx), int(gy)), 5)
        pygame.draw.circle(surf, (220, 255, 230), (int(gx), int(gy)), 2)

    def _draw_path_map(self, surf: pygame.Surface, path: list[tuple[float, float]]) -> None:
        if len(path) < 2:
            return
        # Draw path with subtle glow
        glow_surf = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
        pygame.draw.lines(glow_surf, (*PATH, 40), False, path, 7)
        surf.blit(glow_surf, (0, 0))
        pygame.draw.lines(surf, PATH, False, path, 2)

    def _draw_slam_view(self, surf: pygame.Surface, state: SimState, grid: np.ndarray, resolution: float) -> None:
        surf.fill(MAP_BG)
        self._draw_grid(surf, grid, resolution)
        for obs in state.world.dynamic_obstacles:
            pygame.draw.rect(surf, DYNAMIC_OBSTACLE, pygame.Rect(obs.x, obs.y, obs.w, obs.h), border_radius=4)
        self._draw_goal_map(surf, state)
        self._draw_path_map(surf, state.path)
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
        show_help: bool = False,
    ) -> None:
        self._update_camera(state)
        left_rect, right_rect = self._draw_frame(screen)
        world_surface = pygame.Surface((self.width, self.height))
        map_surface = pygame.Surface((self.width, self.height))

        if self.driving_view == "isometric":
            self._draw_world_iso(world_surface, state)
            self._draw_goal_iso(world_surface, state)
            self._draw_path_iso(world_surface, state, state.path)
            self._draw_lidar_iso(world_surface, state, lidar)
            self._draw_car_iso(world_surface, state)
        elif self.driving_view == "chase":
            self._draw_world_chase(world_surface, state)
            self._draw_goal_chase(world_surface, state)
            self._draw_path_chase(world_surface, state, state.path)
            self._draw_lidar_chase(world_surface, state, lidar)
            self._draw_car_chase(world_surface)
        else:
            self._draw_world(world_surface, state)
            self._draw_goal(world_surface, state)
            self._draw_path(world_surface, state.path)
            self._draw_lidar(world_surface, state, lidar)
            self._draw_car(world_surface, state)
        self._draw_hud(world_surface, mode, state.t, state.collided)
        if show_help:
            self._draw_controls(world_surface)

        self._draw_slam_view(map_surface, state, grid, grid_resolution)
        self._draw_hud(map_surface, mode, state.t, state.collided)

        screen.blit(world_surface, left_rect.topleft)
        screen.blit(map_surface, right_rect.topleft)

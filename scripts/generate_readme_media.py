from __future__ import annotations

import os
from pathlib import Path

import pygame

from drivesim.ml.env import DriveSimEnv, EnvConfig
from drivesim.ui.render import Renderer2D


OUT_DIR = Path("imgs/readme")


def _prepare_env(mapping_mode: str, map_name: str, seed: int) -> tuple[DriveSimEnv, dict]:
    env = DriveSimEnv(
        EnvConfig(
            map_name=map_name,
            auto_expand=False,
            dynamic_obstacle_count=1,
            seed=seed,
            mapping_mode=mapping_mode,
        )
    )
    obs = env.reset(seed=seed)
    state = env.sim.get_state()
    for _ in range(35):
        action = env.autopilot_action(state)
        obs, _, done, _ = env.step(action)
        state = env.sim.get_state()
        if done:
            break
    return env, obs


def _capture_case(
    *,
    filename: str,
    mapping_mode: str,
    driving_view: str,
    camera_mode: str,
    mode_label: str,
    map_name: str = "default",
    seed: int = 21,
    show_help: bool = False,
) -> None:
    env, obs = _prepare_env(mapping_mode=mapping_mode, map_name=map_name, seed=seed)
    state = env.sim.get_state()
    lidar = env.lidar.read(state)

    renderer = Renderer2D(int(state.world.width), int(state.world.height))
    renderer.set_driving_view(driving_view)
    renderer.set_camera_mode(camera_mode)

    screen = pygame.display.set_mode(renderer.frame_size())
    renderer.render(
        screen,
        state,
        obs["grid"],
        env.mapper.resolution,
        lidar,
        mode_label,
        show_help=show_help,
    )
    pygame.display.flip()
    pygame.image.save(screen, str(OUT_DIR / filename))


def main() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    _capture_case(
        filename="drive_chase_ground_truth.png",
        mapping_mode="ground_truth",
        driving_view="chase",
        camera_mode="follow",
        mode_label="autopilot | map=default | expand=off",
    )
    _capture_case(
        filename="drive_iso_ground_truth.png",
        mapping_mode="ground_truth",
        driving_view="3d",
        camera_mode="cinematic",
        mode_label="assistant (trained:tiny_mlp) | map=default | expand=off",
    )
    _capture_case(
        filename="drive_topdown_ground_truth.png",
        mapping_mode="ground_truth",
        driving_view="topdown",
        camera_mode="tactical",
        mode_label="manual | map=default | expand=off",
        show_help=True,
    )
    _capture_case(
        filename="map_ground_truth.png",
        mapping_mode="ground_truth",
        driving_view="topdown",
        camera_mode="tactical",
        mode_label="autopilot | map=maze | expand=off",
        map_name="maze",
        seed=33,
    )
    _capture_case(
        filename="map_sensor_driven.png",
        mapping_mode="sensor_driven",
        driving_view="topdown",
        camera_mode="tactical",
        mode_label="autopilot | map=maze | expand=off",
        map_name="maze",
        seed=33,
    )
    _capture_case(
        filename="train_live_status.png",
        mapping_mode="sensor_driven",
        driving_view="3d",
        camera_mode="follow",
        mode_label="train-live | fit=4 samples=420 blend=0.31 | last= 96.4 best=142.8 | map=blocks | expand=off",
        map_name="blocks",
        seed=15,
    )
    pygame.quit()
    print(f"generated README media in {OUT_DIR}")


if __name__ == "__main__":
    main()

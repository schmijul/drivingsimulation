from __future__ import annotations

import pygame

from drivesim.core.types import Action
from drivesim.ml.agent import AssistAgent
from drivesim.ml.env import DriveSimEnv
from drivesim.ml.live_train import LivePolicyTrainer
from drivesim.ml.policy import features_from_observation
from drivesim.ml.replay import ReplayLogger
from drivesim.ui.backend import RenderBackend
from drivesim.ui.render import Renderer2D


def _manual_action(keys: pygame.key.ScancodeWrapper) -> Action:
    throttle = 0.0
    steering = 0.0
    if keys[pygame.K_w]:
        throttle += 1.0
    if keys[pygame.K_s]:
        throttle -= 1.0
    if keys[pygame.K_a]:
        steering -= 1.0
    if keys[pygame.K_d]:
        steering += 1.0
    return Action(throttle=throttle, steering=steering)


def run_app() -> None:
    pygame.init()
    env = DriveSimEnv()
    agent = AssistAgent()
    logger = ReplayLogger()

    state = env.sim.get_state()
    renderer: RenderBackend = Renderer2D(int(state.world.width), int(state.world.height))
    screen = pygame.display.set_mode(renderer.frame_size())
    pygame.display.set_caption("DriveSim v0.1")

    clock = pygame.time.Clock()

    mode = "manual"
    modes = ["manual", "autopilot", "assistant", "train-live"]
    maps = env.available_maps()
    current_map = maps.index(env.map_name) if env.map_name in maps else 0
    driving_views = ["chase", "3d", "topdown"]
    current_view = 0
    camera_modes = ["follow", "tactical", "cinematic"]
    current_camera = 0
    show_help = False
    renderer.set_driving_view(driving_views[current_view])
    initial_obs = env.reset()
    trainer = LivePolicyTrainer(LivePolicyTrainer.feature_dim_from_observation(initial_obs))

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_TAB:
                    mode = modes[(modes.index(mode) + 1) % len(modes)]
                elif event.key == pygame.K_r:
                    env.reset()
                elif event.key == pygame.K_c:
                    logger.clear()
                elif event.key == pygame.K_v:
                    current_view = (current_view + 1) % len(driving_views)
                    renderer.set_driving_view(driving_views[current_view])
                elif event.key == pygame.K_f:
                    current_camera = (current_camera + 1) % len(camera_modes)
                    renderer.set_camera_mode(camera_modes[current_camera])
                elif event.key == pygame.K_h:
                    show_help = not show_help
                elif event.key == pygame.K_m:
                    current_map = (current_map + 1) % len(maps)
                    env.set_map(maps[current_map])
                    env.reset()
                    trainer = LivePolicyTrainer(LivePolicyTrainer.feature_dim_from_observation(env._observation(env.sim.get_state())))
                elif event.key == pygame.K_e:
                    env.toggle_auto_expand()

        current = env.sim.get_state()
        obs = env._observation(current)

        if mode == "manual":
            action = _manual_action(pygame.key.get_pressed())
        elif mode == "autopilot":
            action = env.autopilot_action(current)
        elif mode == "train-live":
            action = trainer.act(obs)
        else:
            action = agent.act(obs)

        feature_vec = features_from_observation(obs)
        obs, reward, done, info = env.step(action)
        current = env.sim.get_state()
        if mode == "train-live":
            if trainer.observe(reward, done, obs, info):
                env.reset()
                obs = env._observation(env.sim.get_state())
                current = env.sim.get_state()

        logger.log_step(
            {
                "t": current.t,
                "mode": mode,
                "action": {"throttle": action.throttle, "steering": action.steering},
                "features": feature_vec.tolist(),
                "pose": obs["pose"].tolist(),
                "goal": obs["goal"].tolist(),
                "lidar_front": [float(v) for v in obs["lidar_front"]],
                "reward": float(reward),
                "done": bool(done),
                "distance_to_goal": float(info["distance_to_goal"]),
            }
        )

        lidar_obs = env.lidar.read(current)
        expand_label = "expand=on" if env.auto_expand else "expand=off"
        mode_label = f"{mode} | map={env.map_name} | {expand_label}"
        if mode == "assistant":
            mode_label = f"{agent.mode_label} | map={env.map_name} | {expand_label}"
        elif mode == "train-live":
            mode_label = f"{trainer.status_label()} | map={env.map_name} | {expand_label}"
        renderer.render(screen, current, obs["grid"], env.mapper.resolution, lidar_obs, mode_label, show_help=show_help)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

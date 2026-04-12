# Architecture

## Overview

DriveSim is organized around four layers:

- `core`: world state, vehicle state, simulator time-step logic, and scenario construction
- `autonomy`: sensors, occupancy mapping, path planning, and path following
- `ml`: environment wrappers, model loading, feature extraction, training, evaluation, and experiment history
- `ui`: interactive pygame application and renderer backend

The main interactive entry point is [`run_app`](../src/drivesim/ui/app.py#L32). The packaging entry point `drivesim-run` is declared in [`pyproject.toml`](../pyproject.toml).

## Runtime Data Flow

The standard closed loop in `autopilot` and `assistant` mode is:

1. The simulator advances vehicle and world state in [`Simulator`](../src/drivesim/core/simulator.py).
2. A lidar scan is produced by [`LidarSensor.read`](../src/drivesim/autonomy/sensors.py#L36).
3. The occupancy grid is updated by [`OccupancyGridMapper.update`](../src/drivesim/autonomy/mapping.py#L60).
4. A path on the grid is planned by [`AStarPlanner.plan`](../src/drivesim/autonomy/planner.py#L55).
5. A vehicle command is produced by [`PathController.compute_action`](../src/drivesim/autonomy/controller.py#L15).
6. The UI renders state, grid, and overlays through [`Renderer2D`](../src/drivesim/ui/render.py).

This loop is orchestrated primarily inside [`DriveSimEnv`](../src/drivesim/ml/env.py#L48) and the UI layer in [`run_app`](../src/drivesim/ui/app.py#L32).

## Core Types

The simulator state is intentionally small and explicit:

- [`VehicleState`](../src/drivesim/core/types.py#L9): `x`, `y`, `yaw`, `speed`, `steering`
- [`Action`](../src/drivesim/core/types.py#L18): `throttle`, `steering`
- [`Obstacle`](../src/drivesim/core/types.py#L24): axis-aligned rectangle
- [`DynamicObstacle`](../src/drivesim/core/types.py#L32): obstacle plus `vx`, `vy`
- [`World`](../src/drivesim/core/types.py#L38): map bounds, obstacles, start/goal
- [`SimState`](../src/drivesim/core/types.py#L48): vehicle, collision flag, time, world, current path

These dataclasses are the glue between the simulator, autonomy stack, training code, and UI.

## Environment Layer

[`DriveSimEnv`](../src/drivesim/ml/env.py#L48) is the most important integration point. It owns:

- scenario selection and reset logic
- lidar sensor and occupancy mapper construction
- A* planner and path controller construction
- dynamic obstacle spawning
- optional world expansion
- observation generation
- reward and termination logic

The corresponding config object is [`EnvConfig`](../src/drivesim/ml/env.py#L35). This is the right place to look if you want to understand episode behavior or change rollout defaults.

For RL frameworks, the Gym-compatible wrapper lives in [`DriveSimGymEnv`](../src/drivesim/ml/env.py).

## Model Layer

Model loading is centralized in [`load_policy_model`](../src/drivesim/ml/models.py#L145). The project currently supports:

- linear policies stored as `.npz`
- compact `tiny_mlp` policies stored as `.npz`
- PPO checkpoints stored as `.zip`

Feature extraction for non-RL policies is implemented in [`features_from_observation`](../src/drivesim/ml/policy.py#L69). This design keeps model serialization simple and makes the observation-to-feature contract explicit.

## UI Layer

The UI bootstrap lives in [`run_app`](../src/drivesim/ui/app.py#L32). It wires together:

- `DriveSimEnv`
- [`AssistAgent`](../src/drivesim/ml/agent.py)
- [`ReplayLogger`](../src/drivesim/ml/replay.py)
- [`LivePolicyTrainer`](../src/drivesim/ml/live_train.py)
- the renderer backend

The UI mode switch is implemented by the `mode` variable in [`src/drivesim/ui/app.py`](../src/drivesim/ui/app.py#L49). That one variable decides whether actions come from manual control, autopilot, assistant inference, or live teacher-guided learning.

## Tests as Executable Documentation

The tests are worth reading as design references:

- [`tests/test_planner.py`](../tests/test_planner.py): path planner behavior
- [`tests/test_policy.py`](../tests/test_policy.py): feature extraction and policy math
- [`tests/test_train_anyone.py`](../tests/test_train_anyone.py): BC+DAgger workflow expectations
- [`tests/test_eval.py`](../tests/test_eval.py): evaluation summary semantics
- [`tests/test_env.py`](../tests/test_env.py): environment behavior and reset dynamics

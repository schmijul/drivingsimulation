# Documentation

This directory contains the detailed project documentation. The top-level `README.md` is intentionally kept as an entry point; detailed explanations for algorithms, variables, APIs, and workflows live here.

## Pages

- [Architecture](architecture.md): module boundaries, runtime flow, and ownership of major components
- [Algorithms](algorithms.md): lidar sensing, occupancy mapping, path planning, path following, and training algorithms
- [API reference](api.md): public runtime entry points, observation schema, CLI commands, and extension points
- [Variables and configuration](variables.md): important constants, configuration fields, environment variables, and persisted artifacts
- [Training and evaluation](training-and-evaluation.md): replay, BC+DAgger, PPO, live training, and evaluation workflows

## Source Anchors

Start here if you want to read implementation first:

- [`DriveSimEnv`](../src/drivesim/ml/env.py)
- [`AStarPlanner`](../src/drivesim/autonomy/planner.py)
- [`OccupancyGridMapper`](../src/drivesim/autonomy/mapping.py)
- [`PathController`](../src/drivesim/autonomy/controller.py)
- [`LinearPolicy` feature extraction](../src/drivesim/ml/policy.py)
- [`train_anyone`](../src/drivesim/ml/train_anyone.py)
- [`run_eval`](../src/drivesim/ml/eval.py)
- [`run_app`](../src/drivesim/ui/app.py)

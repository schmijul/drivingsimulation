# DriveSim

DriveSim is a stylized 2D driving and SLAM simulator with an ML-ready API.

## Features (v0.1)
- Deterministic 2D simulator with vehicle kinematics and collision handling
- Lidar-style raycast sensing
- Occupancy-grid mapping as a SLAM building block
- A* path planning and a lightweight path-following controller
- Gym-like environment API (`reset`, `step`)
- Stylized Pygame visualization for demo and debugging
- Episode replay logging as JSONL

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
drivesim-run
```

Viewer controls:
- `W/S`: throttle / brake
- `A/D`: steer
- `TAB`: switch mode (`manual` / `autopilot` / `assistant`)
- `R`: reset
- `ESC`: quit

## Architecture
- `drivesim.core`: world model, vehicle, simulation loop
- `drivesim.autonomy`: sensing, mapping, planning, control
- `drivesim.ml`: gym-like env, assistant agent, replay logger
- `drivesim.ui`: renderer and interaction layer

## Tests

```bash
pytest
```

## Next steps
- Add a 3D renderer backend (for example Panda3D or a Unity bridge)
- Train an assistant policy from replay data
- Add dynamic obstacles and scenario generation

## Make targets
- `make install-dev`: install editable package with dev dependencies
- `make run`: start the simulator UI
- `make test`: run test suite

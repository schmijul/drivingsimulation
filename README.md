# DriveSim

DriveSim is a stylized 2D driving and autonomy simulator with an ML-ready API.

## Current UI Snapshots

![DriveSim 2D split view](imgs/2dview.png)
![DriveSim isometric split view](imgs/3dview.png)

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
drivesim-run
```

Install RL extras for Gym + SB3 workflows:

```bash
pip install -e .[dev,rl]
```

Useful shortcuts:

```bash
make run
make test
make train-anyone
make eval
```

## Viewer Controls

- `W/S`: throttle / brake
- `A/D`: steer
- `TAB`: switch mode (`manual` / `autopilot` / `assistant` / `train-live`)
- `R`: reset
- `M`: switch map preset
- `V`: switch driving camera (`chase` / `3d` / `topdown`)
- `F`: switch camera behavior (`follow` / `tactical` / `cinematic`)
- `E`: toggle auto-expanding world
- `P`: save current best live-trained model to `models/assist_policy.npz`
- `H`: toggle controls overlay
- `C`: clear replay log (`replays/latest_episode.jsonl`)
- `ESC`: quit

## Documentation

- [Documentation index](docs/index.md)
- [Architecture](docs/architecture.md)
- [Algorithms](docs/algorithms.md)
- [API reference](docs/api.md)
- [Variables and configuration](docs/variables.md)
- [Training and evaluation](docs/training-and-evaluation.md)

## Common Commands

```bash
DRIVESIM_START_VIEW=3d drivesim-run
DRIVESIM_START_MODE=assistant DRIVESIM_START_VIEW=3d drivesim-run
DRIVESIM_MAPPING_MODE=sensor_driven drivesim-run
make docs-media
make test-workflows
```

## Code Layout

- `src/drivesim/core`: vehicle, world, simulator primitives
- `src/drivesim/autonomy`: sensing, mapping, planning, control
- `src/drivesim/ml`: environment wrappers, policies, training, evaluation
- `src/drivesim/ui`: renderer and interactive application
- `tests`: unit and workflow coverage

## Make Targets

- `make install-dev`: install editable package with dev dependencies
- `make install-rl`: install editable package with RL dependencies
- `make run`: start the simulator UI
- `make test`: run the full test suite
- `make test-workflows`: run workflow-focused tests
- `make train`: start visible `train-live` mode
- `make train-anyone`: run the BC+DAgger trainer
- `make train-auto`: run headless self-training
- `make train-rl`: run PPO training
- `make eval`: run headless evaluation
- `make eval-compare`: compare assistant vs autopilot
- `make eval-report`: write timestamped JSON evaluation reports
- `make eval-history`: show recent evaluation history
- `make eval-best`: show the best recorded evaluation row
- `make experiment-history`: show tracked training/eval runs

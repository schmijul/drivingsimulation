# Variables and Configuration

## Why This Page Exists

This project has relatively few global constants, but a lot of behavior is controlled by dataclass fields, constructor defaults, and startup environment variables. This page collects the important ones and points to the code that owns them.

## Environment Variables

Startup environment variables are read in [`run_app`](../src/drivesim/ui/app.py#L32).

- `DRIVESIM_MAPPING_MODE`
  Used at startup in [`src/drivesim/ui/app.py`](../src/drivesim/ui/app.py#L34). Accepted values: `ground_truth`, `sensor_driven`.
- `DRIVESIM_START_MODE`
  Used in [`src/drivesim/ui/app.py`](../src/drivesim/ui/app.py#L65). Accepted values: `manual`, `autopilot`, `assistant`, `train-live`.
- `DRIVESIM_START_VIEW`
  Used in [`src/drivesim/ui/app.py`](../src/drivesim/ui/app.py#L68). Accepted values: `chase`, `3d`, `topdown`.
- `DRIVESIM_START_CAMERA`
  Used in [`src/drivesim/ui/app.py`](../src/drivesim/ui/app.py#L72). Accepted values: `follow`, `tactical`, `cinematic`.
- `DRIVESIM_ASSIST_MODEL`
  Used by [`AssistAgent`](../src/drivesim/ml/agent.py) to override the default model path.

## `EnvConfig`

[`EnvConfig`](../src/drivesim/ml/env.py#L35) controls episode and environment behavior.

- `max_steps`
  Hard episode length cap.
- `map_name`
  Initial scenario name.
- `auto_expand`
  Enables chunk-based world growth.
- `chunk_size`
  Size of newly generated world chunks.
- `expand_margin`
  Distance from the world edge that triggers expansion.
- `dynamic_obstacle_count`
  Number of moving rectangular obstacles to spawn.
- `dynamic_obstacle_speed`
  Upper speed scale used when spawning dynamic obstacles.
- `seed`
  Base RNG seed.
- `mapping_mode`
  `ground_truth` or `sensor_driven`.

Runtime effects can be traced in [`DriveSimEnv.__init__`](../src/drivesim/ml/env.py#L49), [`randomize_episode`](../src/drivesim/ml/env.py#L125), and [`_spawn_dynamic_obstacles`](../src/drivesim/ml/env.py#L177).

## Mapping Variables

[`OccupancyGridMapper`](../src/drivesim/autonomy/mapping.py#L9) has a few variables that strongly affect behavior:

- `resolution=8.0`
  Larger values make planning coarser and faster; smaller values increase detail and grid size.
- `mapping_mode`
  Chooses baked obstacle truth versus online sensor accumulation.
- `grid`
  The mutable occupancy tensor with values in `[0.0, 1.0]`.
- `rows`, `cols`
  Derived from world dimensions and resolution.

Evidence update constants:

- `mark_free`: subtract `0.25` in [`src/drivesim/autonomy/mapping.py`](../src/drivesim/autonomy/mapping.py#L36)
- `mark_occupied`: add `0.45` in [`src/drivesim/autonomy/mapping.py`](../src/drivesim/autonomy/mapping.py#L40)

## Planner Variables

[`AStarPlanner`](../src/drivesim/autonomy/planner.py#L12) exposes the main planning constants:

- `occupancy_threshold=0.6`
  Above this value a cell is blocked.
- `obstacle_cost_scale=4.0`
  Below-threshold occupancy still increases path cost.
- `smooth_path=True`
  Enables post-search collinearity compression.

The actual use sites are:

- blocked-cell check in [`plan`](../src/drivesim/autonomy/planner.py#L55)
- soft cost in [`_cell_cost`](../src/drivesim/autonomy/planner.py#L36)
- smoothing in [`_smooth_path`](../src/drivesim/autonomy/planner.py#L40)

## Controller Variables

[`PathController`](../src/drivesim/autonomy/controller.py#L11) is governed mainly by:

- `lookahead=6`
  Index into the planned path used as steering target.
- steering normalization divisor `0.8`
  Used in [`compute_action`](../src/drivesim/autonomy/controller.py#L25) to map heading error to steering.
- throttle levels `0.75`, `0.35`, `0.2`
  Used in [`compute_action`](../src/drivesim/autonomy/controller.py#L26) through [`src/drivesim/autonomy/controller.py`](../src/drivesim/autonomy/controller.py#L29).

These values are heuristic, not learned.

## Feature Variables for Learned Policies

The learned non-RL policies depend on the feature contract in [`features_from_observation`](../src/drivesim/ml/policy.py#L69).

Primary derived variables:

- `goal_dx`, `goal_dy`
- `goal_dist`
- `goal_forward`
- `goal_lateral`
- `heading_error`
- sampled lidar distances

Normalization variables saved with a trained model:

- `feature_mean`
- `feature_std`

Trainable parameters of the linear model:

- `weights`
- `bias`

See [`LinearPolicy`](../src/drivesim/ml/policy.py#L12).

## `TrainAnyoneConfig`

[`TrainAnyoneConfig`](../src/drivesim/ml/train_anyone.py#L15) controls the default BC+DAgger trainer.

Data collection:

- `maps`
- `max_steps`
- `min_goal_distance`
- `dynamic_obstacle_count`
- `bc_episodes_per_map`
- `dagger_rounds`
- `dagger_episodes_per_map`

Teacher blending:

- `teacher_mix_start`
- `teacher_mix_end`

Model and optimization:

- `architecture`
- `l2_reg`
- `mlp_hidden_dim`
- `mlp_epochs`
- `mlp_lr`
- `mlp_batch_size`

Evaluation after training:

- `eval_episodes_per_map`
- `eval_dynamic_obstacles`

These values are applied in [`train_anyone`](../src/drivesim/ml/train_anyone.py#L158) and its helper functions.

## `EvalConfig`

[`EvalConfig`](../src/drivesim/ml/eval.py#L19) controls evaluation runs.

- `maps`
- `episodes_per_map`
- `max_steps`
- `seed`
- `policy_mode`
- `model_path`
- `dynamic_obstacle_count`
- `mapping_mode`
- `min_goal_distance`
- `curriculum`
- `rl_algorithm`
- `json_out`
- `track_run`
- `experiment_history_path`

These variables affect curriculum construction, environment setup, JSON report writing, and experiment tracking inside [`run_eval`](../src/drivesim/ml/eval.py#L202).

## Artifact Paths

Common path variables used throughout the project:

- `models/assist_policy.npz`
  Main assistant checkpoint path.
- `models/rl/<run>/model.zip`
  PPO checkpoint path.
- `replays/latest_episode.jsonl`
  Replay log used by replay training and debugging.
- `replays/evals/index.jsonl`
  Compact evaluation history.
- `replays/experiments/index.jsonl`
  Training and evaluation experiment history.

If you want to change persistence conventions, start in:

- [`src/drivesim/ml/replay.py`](../src/drivesim/ml/replay.py)
- [`src/drivesim/ml/eval.py`](../src/drivesim/ml/eval.py)
- [`src/drivesim/ml/experiment.py`](../src/drivesim/ml/experiment.py)
- [`src/drivesim/ml/models.py`](../src/drivesim/ml/models.py)

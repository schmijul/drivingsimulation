# Algorithms

## Lidar Raycasting

The simulated lidar is implemented in [`LidarSensor`](../src/drivesim/autonomy/sensors.py#L17).

Relevant parameters:

- `rays=41`
- `fov=180 degrees`
- `max_range=120.0`
- `step=2.0`

Algorithm outline from [`read`](../src/drivesim/autonomy/sensors.py#L36):

1. Generate evenly spaced ray offsets across the configured field of view.
2. For each ray, march forward in `step` increments.
3. Stop when the ray hits world bounds, a static obstacle, or a dynamic obstacle.
4. Return the hit distance, clipped by `max_range`.

The occupancy test is isolated in [`_occupied`](../src/drivesim/autonomy/sensors.py#L24). This keeps collision geometry simple: all obstacles are treated as axis-aligned rectangles.

## Occupancy Mapping

The occupancy grid is implemented by [`OccupancyGridMapper`](../src/drivesim/autonomy/mapping.py#L9).

Two mapping modes are supported:

- `ground_truth`: static world obstacles are baked into the grid at initialization via [`bake_world_obstacles`](../src/drivesim/autonomy/mapping.py#L53)
- `sensor_driven`: the grid starts empty and is built from sensor updates over time

Important details:

- Grid resolution defaults to `8.0` world units per cell in [`__init__`](../src/drivesim/autonomy/mapping.py#L10).
- Free-space evidence decays a cell by `0.25` in [`mark_free`](../src/drivesim/autonomy/mapping.py#L36).
- Occupied-space evidence increases a cell by `0.45` in [`mark_occupied`](../src/drivesim/autonomy/mapping.py#L40).
- Values are clamped to `[0.0, 1.0]`.

Update logic in [`update`](../src/drivesim/autonomy/mapping.py#L60):

1. For each lidar ray, traverse intermediate samples and mark them free.
2. Mark the ray endpoint occupied.
3. Return the current grid.

This is a deliberately lightweight inverse sensor model. It is easy to reason about and fast enough for live rendering and repeated training rollouts.

## A* Path Planning

Path planning is implemented in [`AStarPlanner`](../src/drivesim/autonomy/planner.py#L12).

Important parameters from [`__init__`](../src/drivesim/autonomy/planner.py#L13):

- `occupancy_threshold=0.6`: cells at or above this value are treated as blocked
- `obstacle_cost_scale=4.0`: soft obstacle penalty multiplier
- `smooth_path=True`: compress collinear segments after search

Algorithm details:

- Neighbor expansion is 8-connected in [`_neighbors`](../src/drivesim/autonomy/planner.py#L18).
- The heuristic is Euclidean distance in [`_h`](../src/drivesim/autonomy/planner.py#L32).
- Soft costs come from occupancy probability times `obstacle_cost_scale` in [`_cell_cost`](../src/drivesim/autonomy/planner.py#L36).
- Path reconstruction and optional smoothing happen in [`plan`](../src/drivesim/autonomy/planner.py#L55).

The planner therefore combines hard obstacle rejection with soft risk aversion. Even when a cell is still below the blocking threshold, a high occupancy value makes the route less attractive.

## Path Following Controller

The local controller is implemented in [`PathController`](../src/drivesim/autonomy/controller.py#L11).

Key behavior:

- It targets the waypoint at index `lookahead`, default `6`, from [`__init__`](../src/drivesim/autonomy/controller.py#L12).
- Heading error is normalized to `[-pi, pi]` in [`compute_action`](../src/drivesim/autonomy/controller.py#L15).
- Steering is proportional to heading error with clipping to `[-1.0, 1.0]`.
- Throttle is reduced when the heading error is large.

This is intentionally a simple geometric controller. It is not a bicycle-model MPC or pure pursuit implementation with curvature optimization. The benefit is predictability and low maintenance cost.

## Feature Engineering for Learned Policies

Non-RL policies consume engineered features from [`features_from_observation`](../src/drivesim/ml/policy.py#L69).

The default feature vector contains:

- current speed
- goal position projected into vehicle frame: `goal_forward`, `goal_lateral`
- scalar goal distance
- heading error to the goal
- 9 sampled lidar distances

Why the vehicle-frame projection matters:

- It reduces dependence on absolute map coordinates.
- It helps transfer between maps and randomized starts.
- It makes the policy learn steering relative to the car, not the screen.

Legacy 12-dimensional features remain supported through [`features_for_dim`](../src/drivesim/ml/policy.py#L108) for backward compatibility with older saved models.

## Linear Policy Fitting

The linear model is trained by [`fit_linear_policy`](../src/drivesim/ml/policy.py#L121).

The fitting procedure is ridge regression:

1. Compute per-feature mean and standard deviation.
2. Normalize the design matrix.
3. Append a bias column of ones.
4. Solve the regularized normal equation.
5. Store `weights`, `bias`, `feature_mean`, and `feature_std`.

This gives a compact model with transparent parameters and deterministic inference.

## BC + DAgger Trainer

The default trainer for first-time users is implemented in [`train_anyone`](../src/drivesim/ml/train_anyone.py#L158).

Workflow:

1. Collect teacher demonstrations with autopilot only in [`_collect_supervised_rows`](../src/drivesim/ml/train_anyone.py#L47).
2. Fit an initial policy.
3. Run DAgger rounds where rollout actions are blended between teacher and current model using [`_blend_action`](../src/drivesim/ml/train_anyone.py#L38).
4. Append newly collected states with teacher labels.
5. Refit the model on the aggregated dataset.
6. Evaluate the final model with [`run_eval`](../src/drivesim/ml/eval.py#L202).

Relevant variables:

- `teacher_mix_start` and `teacher_mix_end` in [`TrainAnyoneConfig`](../src/drivesim/ml/train_anyone.py#L15)
- `bc_episodes_per_map`
- `dagger_rounds`
- `dagger_episodes_per_map`

The design goal is robustness rather than benchmark-optimal sample efficiency.

## Tiny MLP Optimization

The compact neural policy is trained in [`_fit_tiny_mlp`](../src/drivesim/ml/train_anyone.py#L87).

Implementation details:

- one hidden layer with `tanh`
- manual forward pass and backprop in NumPy
- mini-batch updates
- small L2 stabilizer applied to both weight matrices

This keeps the training stack lightweight and removes the need for a heavy deep-learning dependency in the default path.

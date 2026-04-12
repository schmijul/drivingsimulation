# API Reference

## Entry Points

Package scripts are declared in [`pyproject.toml`](../pyproject.toml):

- `drivesim-run`: interactive UI via [`drivesim.main:main`](../src/drivesim/main.py#L4)
- `drivesim-train`: replay trainer via [`drivesim.ml.train:main`](../src/drivesim/ml/train.py)
- `drivesim-train-rl`: PPO trainer via [`drivesim.ml.train_rl:main`](../src/drivesim/ml/train_rl.py)
- `drivesim-train-anyone`: BC+DAgger trainer via [`drivesim.ml.train_anyone:main`](../src/drivesim/ml/train_anyone.py#L241)
- `drivesim-train-auto`: headless self-training via [`drivesim.ml.train_auto:main`](../src/drivesim/ml/train_auto.py)
- `drivesim-eval`: evaluator via [`drivesim.ml.eval:main`](../src/drivesim/ml/eval.py)
- `drivesim-eval-history`: evaluation history browser via [`drivesim.ml.eval_history:main`](../src/drivesim/ml/eval_history.py)
- `drivesim-experiment-history`: experiment history browser via [`drivesim.ml.experiment_history:main`](../src/drivesim/ml/experiment_history.py)

The `Makefile` wraps the most common commands in a shorter interface. See [`Makefile`](../Makefile).

## `DriveSimEnv`

[`DriveSimEnv`](../src/drivesim/ml/env.py#L48) is the main programmatic environment API.

Core methods:

- [`available_maps`](../src/drivesim/ml/env.py#L73): list map names
- [`set_map`](../src/drivesim/ml/env.py#L76): rebuild world for a named scenario
- [`set_mapping_mode`](../src/drivesim/ml/env.py#L89): switch between `ground_truth` and `sensor_driven`
- [`toggle_auto_expand`](../src/drivesim/ml/env.py#L96): enable/disable chunked map growth
- [`randomize_episode`](../src/drivesim/ml/env.py#L125): sample a new reachable start and goal
- [`reset`](../src/drivesim/ml/env.py#L280): reset state and mapper, optionally with a new seed
- [`autopilot_action`](../src/drivesim/ml/env.py#L326): compute a controller action from the autonomy stack
- [`step`](../src/drivesim/ml/env.py#L332): advance the simulation by one action

### Observation schema

Observations are assembled in [`_observation`](../src/drivesim/ml/env.py#L302). Keys:

- `pose`: `np.array([x, y, yaw, speed], dtype=np.float32)`
- `goal`: `np.array([goal_x, goal_y], dtype=np.float32)`
- `grid`: occupancy grid as `np.ndarray`
- `lidar`: full lidar distance vector
- `lidar_front`: five central lidar values for replay logs and HUD-friendly summaries
- `collided`: boolean collision state
- `map_name`: current map identifier

This observation contract is consumed by:

- [`LinearPolicy.act`](../src/drivesim/ml/policy.py#L23)
- [`TinyMLPPolicyModel.act`](../src/drivesim/ml/models.py#L37)
- [`PPOPolicyModel._flat_observation`](../src/drivesim/ml/models.py#L76)
- [`AssistAgent`](../src/drivesim/ml/agent.py)

### Step return contract

[`step`](../src/drivesim/ml/env.py#L332) returns `(obs, reward, done, info)`.

Important `info` values used elsewhere in the project:

- `distance_to_goal`
- `goal_reached`
- `map_name`
- `collided`

See consumers in [`run_eval`](../src/drivesim/ml/eval.py#L202) and [`run_app`](../src/drivesim/ui/app.py#L128).

## Config Objects

Primary dataclass configs:

- [`EnvConfig`](../src/drivesim/ml/env.py#L35): map, seed, mapping mode, expansion, obstacle counts, max steps
- [`EvalConfig`](../src/drivesim/ml/eval.py#L19): evaluation maps, seeds, model path, curriculum, JSON reporting
- [`TrainAnyoneConfig`](../src/drivesim/ml/train_anyone.py#L15): BC+DAgger parameters and tiny MLP hyperparameters

These dataclasses are the best reference when you want to know which knobs actually exist.

## Policy Model API

The common inference protocol is defined by [`PolicyModel`](../src/drivesim/ml/models.py#L14). All policy models expose:

- `model_name`
- `act(observation) -> Action`

Concrete implementations:

- [`LinearPolicyModel`](../src/drivesim/ml/models.py#L21)
- [`TinyMLPPolicyModel`](../src/drivesim/ml/models.py#L30)
- [`PPOPolicyModel`](../src/drivesim/ml/models.py#L71)

Model loading is centralized in [`load_policy_model`](../src/drivesim/ml/models.py#L145). If you add a new persisted policy type, this function is the integration point.

## UI Runtime API

The interactive application is not a public library API in the strict sense, but these hooks are operationally important:

- [`run_app`](../src/drivesim/ui/app.py#L32): bootstrap the simulator UI
- [`_manual_action`](../src/drivesim/ui/app.py#L18): keyboard input mapping
- environment variables read at startup in [`src/drivesim/ui/app.py`](../src/drivesim/ui/app.py#L34)

Supported startup environment variables:

- `DRIVESIM_MAPPING_MODE`
- `DRIVESIM_START_MODE`
- `DRIVESIM_START_VIEW`
- `DRIVESIM_START_CAMERA`

## Persistence Formats

Important project artifacts:

- replay log: `replays/latest_episode.jsonl` written by [`ReplayLogger`](../src/drivesim/ml/replay.py)
- assistant model: `models/assist_policy.npz`
- RL checkpoints: `models/rl/<run>/model.zip`
- eval history: `replays/evals/index.jsonl`
- experiment history: `replays/experiments/index.jsonl`

Serialization logic is implemented in:

- [`LinearPolicy.save`](../src/drivesim/ml/policy.py#L30)
- [`TinyMLPPolicyModel.save`](../src/drivesim/ml/models.py#L46)
- [`_write_eval_json`](../src/drivesim/ml/eval.py#L62)

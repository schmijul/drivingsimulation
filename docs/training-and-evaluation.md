# Training and Evaluation

## Training Modes

The project currently supports four distinct training paths:

- replay training via [`src/drivesim/ml/train.py`](../src/drivesim/ml/train.py)
- visible live training via [`LivePolicyTrainer`](../src/drivesim/ml/live_train.py)
- BC+DAgger via [`train_anyone`](../src/drivesim/ml/train_anyone.py#L158)
- PPO via [`src/drivesim/ml/train_rl.py`](../src/drivesim/ml/train_rl.py)

The command routing for the common developer interface lives in [`Makefile`](../Makefile).

## Replay Training

Replay training learns from the JSONL replay log produced by the UI.

Main code:

- replay logger in [`src/drivesim/ml/replay.py`](../src/drivesim/ml/replay.py)
- replay parsing in [`src/drivesim/ml/train.py`](../src/drivesim/ml/train.py)
- model fit in [`fit_linear_policy`](../src/drivesim/ml/policy.py#L121)

Typical flow:

1. Drive in `manual` or `autopilot` mode.
2. Collect `replays/latest_episode.jsonl`.
3. Run `make train` or `python -m drivesim.ml.train`.
4. Load the resulting `models/assist_policy.npz` in `assistant` mode.

## Live Training

Live online training is integrated into the UI loop in [`run_app`](../src/drivesim/ui/app.py#L120).

Relevant pieces:

- trainer construction in [`src/drivesim/ui/app.py`](../src/drivesim/ui/app.py#L60)
- teacher observation in [`src/drivesim/ui/app.py`](../src/drivesim/ui/app.py#L121)
- online update trigger in [`src/drivesim/ui/app.py`](../src/drivesim/ui/app.py#L130)
- trainer implementation in [`src/drivesim/ml/live_train.py`](../src/drivesim/ml/live_train.py)

Design intent:

- start from an existing linear model if available
- keep the teacher in the loop so the car does not spend most of its time spinning or crashing
- allow quick interactive iteration without leaving the viewer

## BC + DAgger

The recommended default path is `make train-anyone`.

Implementation reference:

- config dataclass in [`TrainAnyoneConfig`](../src/drivesim/ml/train_anyone.py#L15)
- data collection in [`_collect_supervised_rows`](../src/drivesim/ml/train_anyone.py#L47)
- tiny MLP fitting in [`_fit_tiny_mlp`](../src/drivesim/ml/train_anyone.py#L87)
- full workflow in [`train_anyone`](../src/drivesim/ml/train_anyone.py#L158)

The trainer always ends with an evaluation pass through [`run_eval`](../src/drivesim/ml/eval.py#L202), so every training run produces a direct quality signal.

## PPO Training

The RL-specific path is implemented in [`src/drivesim/ml/train_rl.py`](../src/drivesim/ml/train_rl.py).

Dependencies:

- `gymnasium`
- `stable-baselines3`

Support utilities:

- RL dependency loading in [`src/drivesim/ml/rl_utils.py`](../src/drivesim/ml/rl_utils.py)
- flattened PPO observation adapter in [`PPOPolicyModel._flat_observation`](../src/drivesim/ml/models.py#L76)

This path is for users who want a standard Gym-compatible interface and SB3 integration rather than the lightweight NumPy-only default trainers.

## Evaluation

Evaluation is implemented in [`run_eval`](../src/drivesim/ml/eval.py#L202).

Main metrics from [`_summarize`](../src/drivesim/ml/eval.py#L186):

- `success_rate`
- `collision_rate`
- `avg_distance_to_goal`
- `avg_steps`
- `avg_total_reward`

Two evaluation backends exist:

- assistant/autopilot path through [`_episode_metrics`](../src/drivesim/ml/eval.py#L114)
- PPO path through [`_episode_metrics_rl`](../src/drivesim/ml/eval.py#L153)

The evaluator can also:

- run a curriculum
- compare policies
- write JSON reports
- append compact history rows
- track experiment metadata

Related functions:

- [`default_eval_report_path`](../src/drivesim/ml/eval.py#L34)
- [`_append_eval_history`](../src/drivesim/ml/eval.py#L40)
- [`_write_eval_json`](../src/drivesim/ml/eval.py#L62)
- experiment tracking helpers in [`src/drivesim/ml/experiment.py`](../src/drivesim/ml/experiment.py)

## History and Reporting

Persisted histories:

- evaluation history in `replays/evals/index.jsonl`
- experiment history in `replays/experiments/index.jsonl`

Query tools:

- [`src/drivesim/ml/eval_history.py`](../src/drivesim/ml/eval_history.py)
- [`src/drivesim/ml/experiment_history.py`](../src/drivesim/ml/experiment_history.py)

Tests covering the reporting pipeline:

- [`tests/test_eval.py`](../tests/test_eval.py)
- [`tests/test_eval_history.py`](../tests/test_eval_history.py)
- [`tests/test_experiment_history.py`](../tests/test_experiment_history.py)

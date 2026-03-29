from pathlib import Path

from drivesim.ml.train_anyone import TrainAnyoneConfig, train_anyone


def test_train_anyone_quick_end_to_end(tmp_path) -> None:
    model_path = tmp_path / "assist_policy.npz"
    _, summary = train_anyone(
        TrainAnyoneConfig(
            maps=["default"],
            seed=5,
            output_path=str(model_path),
            max_steps=120,
            min_goal_distance=120.0,
            dynamic_obstacle_count=0,
            bc_episodes_per_map=2,
            dagger_rounds=1,
            dagger_episodes_per_map=1,
            eval_episodes_per_map=1,
            eval_dynamic_obstacles=0,
            architecture="tiny_mlp",
            mlp_hidden_dim=12,
            mlp_epochs=8,
            mlp_lr=0.01,
            mlp_batch_size=64,
        )
    )
    assert Path(model_path).exists()
    assert "success_rate" in summary
    assert 0.0 <= summary["success_rate"] <= 1.0

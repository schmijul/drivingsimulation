from drivesim.ml.curriculum import list_curricula, resolve_curriculum, single_stage_curriculum


def test_list_curricula_exposes_expected_presets() -> None:
    names = list_curricula()
    assert "easy" in names
    assert "standard" in names
    assert "robust" in names


def test_resolve_curriculum_can_override_maps() -> None:
    curriculum = resolve_curriculum("standard", maps_override=["default"])
    assert curriculum.name == "standard"
    assert all(stage.maps == ("default",) for stage in curriculum.stages)


def test_single_stage_curriculum_uses_custom_defaults() -> None:
    curriculum = single_stage_curriculum(
        maps=["maze", "blocks"],
        dynamic_obstacle_count=2,
        mapping_mode="sensor_driven",
        min_goal_distance=175.0,
    )
    assert curriculum.name == "custom"
    assert curriculum.stages[0].maps == ("maze", "blocks")
    assert curriculum.stages[0].dynamic_obstacle_count == 2
    assert curriculum.stages[0].mapping_mode == "sensor_driven"

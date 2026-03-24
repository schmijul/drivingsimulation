from drivesim.core.scenario import available_scenarios, build_world


def test_generated_maps_are_available() -> None:
    maps = available_scenarios()
    assert "generated_easy" in maps
    assert "generated_medium" in maps
    assert "generated_hard" in maps


def test_generated_map_has_reasonable_world() -> None:
    world = build_world("generated_medium")
    assert world.width >= 800
    assert world.height >= 450
    assert len(world.obstacles) > 0
    assert world.start != world.goal

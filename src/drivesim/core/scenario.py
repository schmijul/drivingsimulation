from drivesim.core.types import Obstacle, World


def default_world() -> World:
    obstacles = [
        Obstacle(120, 90, 60, 220),
        Obstacle(280, 0, 60, 190),
        Obstacle(280, 260, 60, 140),
        Obstacle(420, 120, 90, 60),
        Obstacle(520, 250, 100, 90),
    ]
    return World(
        width=800,
        height=450,
        obstacles=obstacles,
        start=(70.0, 70.0),
        goal=(730.0, 380.0),
    )


def maze_world() -> World:
    obstacles = [
        Obstacle(110, 0, 40, 300),
        Obstacle(220, 150, 40, 300),
        Obstacle(330, 0, 40, 300),
        Obstacle(440, 150, 40, 300),
        Obstacle(550, 0, 40, 300),
    ]
    return World(
        width=800,
        height=450,
        obstacles=obstacles,
        start=(50.0, 40.0),
        goal=(760.0, 405.0),
    )


def blocks_world() -> World:
    obstacles = [
        Obstacle(180, 70, 90, 90),
        Obstacle(350, 220, 120, 80),
        Obstacle(520, 70, 80, 140),
        Obstacle(140, 260, 70, 120),
        Obstacle(610, 280, 120, 70),
    ]
    return World(
        width=800,
        height=450,
        obstacles=obstacles,
        start=(70.0, 380.0),
        goal=(730.0, 70.0),
    )


SCENARIO_BUILDERS = {
    "default": default_world,
    "maze": maze_world,
    "blocks": blocks_world,
}


def available_scenarios() -> list[str]:
    return list(SCENARIO_BUILDERS.keys())


def build_world(name: str) -> World:
    if name not in SCENARIO_BUILDERS:
        name = "default"
    return SCENARIO_BUILDERS[name]()

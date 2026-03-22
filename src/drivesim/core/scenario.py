import random

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


def generate_chunk_obstacles(
    chunk_x: int,
    chunk_y: int,
    chunk_size: int,
    seed: int,
    count: int = 4,
) -> list[Obstacle]:
    rng = random.Random(seed + chunk_x * 92821 + chunk_y * 68917)
    origin_x = chunk_x * chunk_size
    origin_y = chunk_y * chunk_size
    obstacles: list[Obstacle] = []
    for _ in range(count):
        w = float(rng.randint(50, 130))
        h = float(rng.randint(40, 120))
        max_x = origin_x + chunk_size - int(w) - 20
        max_y = origin_y + chunk_size - int(h) - 20
        x = float(rng.randint(origin_x + 20, max(origin_x + 20, max_x)))
        y = float(rng.randint(origin_y + 20, max(origin_y + 20, max_y)))
        obstacles.append(Obstacle(x=x, y=y, w=w, h=h))
    return obstacles

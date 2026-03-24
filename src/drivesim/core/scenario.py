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


def generated_world(difficulty: str, seed: int = 41) -> World:
    width = 900
    height = 540
    start = (70.0, 70.0)
    goal = (width - 70.0, height - 70.0)
    settings = {
        "easy": {"count": 9, "w_min": 55, "w_max": 130, "h_min": 45, "h_max": 120},
        "medium": {"count": 14, "w_min": 60, "w_max": 145, "h_min": 55, "h_max": 130},
        "hard": {"count": 20, "w_min": 68, "w_max": 170, "h_min": 60, "h_max": 150},
    }
    cfg = settings.get(difficulty, settings["medium"])
    rng = random.Random(seed + sum(ord(c) for c in difficulty) * 131)

    obstacles: list[Obstacle] = []
    for _ in range(cfg["count"]):
        w = float(rng.randint(cfg["w_min"], cfg["w_max"]))
        h = float(rng.randint(cfg["h_min"], cfg["h_max"]))
        x = float(rng.randint(20, width - int(w) - 20))
        y = float(rng.randint(20, height - int(h) - 20))
        if abs(x - start[0]) < 100.0 and abs(y - start[1]) < 100.0:
            continue
        if abs(x - goal[0]) < 100.0 and abs(y - goal[1]) < 100.0:
            continue
        obstacles.append(Obstacle(x, y, w, h))

    return World(width=width, height=height, obstacles=obstacles, start=start, goal=goal)


SCENARIO_BUILDERS = {
    "default": default_world,
    "maze": maze_world,
    "blocks": blocks_world,
    "generated_easy": lambda: generated_world("easy"),
    "generated_medium": lambda: generated_world("medium"),
    "generated_hard": lambda: generated_world("hard"),
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

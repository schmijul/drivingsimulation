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

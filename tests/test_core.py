from drivesim.core.scenario import default_world
from drivesim.core.simulator import Simulator
from drivesim.core.types import Action, DynamicObstacle, World


def test_vehicle_moves_forward() -> None:
    sim = Simulator(default_world(), dt=0.1)
    s0 = sim.get_state()
    s1 = sim.step(Action(throttle=1.0, steering=0.0))
    assert s1.vehicle.x > s0.vehicle.x


def test_collision_halts_step() -> None:
    world = default_world()
    sim = Simulator(world, dt=0.1)
    for _ in range(300):
        state = sim.step(Action(throttle=1.0, steering=0.0))
        if state.collided:
            break
    assert state.collided


def test_dynamic_obstacle_moves_each_step() -> None:
    world = World(
        width=200.0,
        height=140.0,
        obstacles=[],
        start=(30.0, 30.0),
        goal=(170.0, 100.0),
        dynamic_obstacles=[DynamicObstacle(x=80.0, y=60.0, w=20.0, h=20.0, vx=40.0, vy=0.0)],
    )
    sim = Simulator(world, dt=0.1)
    x0 = world.dynamic_obstacles[0].x
    sim.step(Action(throttle=0.0, steering=0.0))
    x1 = world.dynamic_obstacles[0].x
    assert x1 > x0


def test_dynamic_obstacle_can_trigger_collision() -> None:
    world = World(
        width=120.0,
        height=120.0,
        obstacles=[],
        start=(50.0, 50.0),
        goal=(100.0, 100.0),
        dynamic_obstacles=[DynamicObstacle(x=46.0, y=46.0, w=12.0, h=12.0, vx=0.0, vy=0.0)],
    )
    sim = Simulator(world, dt=0.1)
    state = sim.step(Action(throttle=0.0, steering=0.0))
    assert state.collided

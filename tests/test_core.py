from drivesim.core.scenario import default_world
from drivesim.core.simulator import Simulator
from drivesim.core.types import Action


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

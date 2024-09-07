import unittest
import pygame
import math
from simulator import Simulator
from car import Car
from obstacle import Obstacle
from radar import Radar

class TestSimulator(unittest.TestCase):
    def setUp(self):
        pygame.init()
        self.simulator = Simulator(800, 600, 5, 0.1, 0.05, 200, 60, 5)

    def tearDown(self):
        pygame.quit()

    def test_simulator_initialization(self):
        self.assertEqual(self.simulator.width, 1600)  # Double the input width
        self.assertEqual(self.simulator.height, 600)
        self.assertIsInstance(self.simulator.car, Car)
        self.assertIsInstance(self.simulator.radar, Radar)
        self.assertEqual(len(self.simulator.obstacles), 5)

    def test_obstacle_spawn(self):
        self.simulator.spawn_initial_obstacles(10)
        self.assertEqual(len(self.simulator.obstacles), 15)  # 5 from init + 10 new
        for obstacle in self.simulator.obstacles:
            self.assertIsInstance(obstacle, Obstacle)
            self.assertTrue(0 <= obstacle.rect.x < self.simulator.width // 2)
            self.assertTrue(0 <= obstacle.rect.y < self.simulator.height)

    def test_collision_detection(self):
        # Place car in a position where it's not colliding
        self.simulator.car.rect.center = (100, 100)
        self.assertFalse(self.simulator.check_collision())

        # Place an obstacle to collide with the car
        colliding_obstacle = Obstacle(90, 90, 30, 30)
        self.simulator.obstacles.append(colliding_obstacle)
        self.assertTrue(self.simulator.check_collision())

class TestCar(unittest.TestCase):
    def setUp(self):
        self.car = Car(400, 300, 5, 0.1, 0.05)

    def test_car_initialization(self):
        self.assertEqual(self.car.rect.center, (400, 300))
        self.assertEqual(self.car.max_speed, 5)
        self.assertEqual(self.car.acceleration, 0.1)
        self.assertEqual(self.car.deceleration, 0.05)

    def test_car_movement(self):
        initial_x = self.car.rect.x
        initial_speed = self.car.speed
        # Simulate pressing the up arrow key
        keys = {pygame.K_UP: True, pygame.K_DOWN: False, pygame.K_LEFT: False, pygame.K_RIGHT: False}
        self.car.move(keys)
        self.assertGreater(self.car.speed, initial_speed)
        # Move multiple times to ensure visible movement
        for _ in range(10):
            self.car.move(keys)
        self.assertGreater(self.car.rect.x, initial_x)

class TestRadar(unittest.TestCase):
    def setUp(self):
        self.radar = Radar(200, 60)
        self.car = Car(400, 300, 5, 0.1, 0.05)
        self.obstacles = [Obstacle(500, 300, 30, 30)]

    def test_radar_initialization(self):
        self.assertEqual(self.radar.range, 200)
        self.assertEqual(self.radar.angle, 60)

    def test_radar_scan(self):
        detections = self.radar.scan(self.car, self.obstacles)
        self.assertGreater(len(detections), 0, "Radar should detect at least one obstacle")
        
        print(f"Number of detections: {len(detections)}")
        print("Detections:", detections)
        
        car_center = self.car.rect.center
        obstacle_center = self.obstacles[0].rect.center
        
        expected_distance = ((obstacle_center[0] - car_center[0])**2 + 
                             (obstacle_center[1] - car_center[1])**2)**0.5
        expected_angle = math.degrees(math.atan2(car_center[1] - obstacle_center[1], 
                                                 obstacle_center[0] - car_center[0]))
        
        print(f"Expected distance: {expected_distance}")
        print(f"Expected angle: {expected_angle}")
        
        # Check if any detection is close to the expected values
        close_detection = any(
            abs(distance - expected_distance) < 20 and abs(angle - expected_angle) < 20
            for distance, angle in detections
        )
        
        self.assertTrue(close_detection, 
                        "At least one detection should be close to the expected values")

        if not close_detection:
            print("No close detections found. Closest detection:")
            closest = min(detections, key=lambda d: abs(d[0] - expected_distance))
            print(f"Distance: {closest[0]}, Angle: {closest[1]}")

if __name__ == '__main__':
    unittest.main()
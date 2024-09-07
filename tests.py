import unittest
import pygame
import numpy as np
import torch
from simulator import Simulator, DQNAgent, Car, Obstacle, Radar, SlamMap

class TestSimulator(unittest.TestCase):
    def setUp(self):
        pygame.init()
        self.simulator = Simulator(800, 600, 5, 0.1, 0.05, 200, 60, 5)

    def tearDown(self):
        pygame.quit()

    def test_simulator_initialization(self):
        self.assertEqual(self.simulator.width, 2400)  # Triple the input width
        self.assertEqual(self.simulator.height, 600)
        self.assertIsInstance(self.simulator.car, Car)
        self.assertIsInstance(self.simulator.radar, Radar)
        self.assertIsInstance(self.simulator.slam_map, SlamMap)
        self.assertEqual(len(self.simulator.obstacles), 5)
        self.assertIsInstance(self.simulator.agent, DQNAgent)

    def test_spawn_goal(self):
        goal = self.simulator.spawn_goal()
        self.assertIsInstance(goal, pygame.Rect)
        self.assertTrue(0 <= goal.x < self.simulator.width // 3)
        self.assertTrue(0 <= goal.y < self.simulator.height)

    def test_spawn_initial_obstacles(self):
        self.simulator.obstacles = []
        self.simulator.spawn_initial_obstacles(3)
        self.assertEqual(len(self.simulator.obstacles), 3)
        for obstacle in self.simulator.obstacles:
            self.assertIsInstance(obstacle, Obstacle)
            self.assertTrue(0 <= obstacle.rect.x < self.simulator.width // 3)
            self.assertTrue(0 <= obstacle.rect.y < self.simulator.height)

    def test_check_collision(self):
        # Clear existing obstacles
        self.simulator.obstacles.clear()

        # Place car in a position where it's not colliding
        self.simulator.car.rect.center = (100, 100)
        self.assertFalse(self.simulator.check_collision())

        # Place an obstacle to collide with the car
        new_obstacle = Obstacle(90, 90, 30, 30)
        self.simulator.obstacles.append(new_obstacle)
        self.assertTrue(self.simulator.check_collision())

        # Move car away from the obstacle
        self.simulator.car.rect.center = (200, 200)
        self.assertFalse(self.simulator.check_collision())

    def test_check_goal_reached(self):
        # Place car away from the goal
        self.simulator.car.rect.center = (100, 100)
        self.simulator.goal.center = (200, 200)
        self.assertFalse(self.simulator.check_goal_reached())

        # Place car on the goal
        self.simulator.car.rect.center = self.simulator.goal.center
        self.assertTrue(self.simulator.check_goal_reached())

    def test_get_state(self):
        state = self.simulator.get_state()
        expected_state_size = self.simulator.slam_map.grid.size + 6
        self.assertEqual(len(state), expected_state_size)

    def test_step(self):
        initial_state = self.simulator.get_state()
        action = 0  # Move forward
        next_state, reward, done = self.simulator.step(action)
        
        self.assertEqual(len(next_state), len(initial_state))
        self.assertIsInstance(reward, (int, float))
        self.assertIsInstance(done, bool)

    def test_slam_map_reset(self):
        # Update SLAM map
        self.simulator.slam_map.update((100, 100), 0, [(50, 0)])
        self.assertTrue(np.any(self.simulator.slam_map.grid != 0))

        # Reset SLAM map
        self.simulator.slam_map.reset()
        self.assertTrue(np.all(self.simulator.slam_map.grid == 0))

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
        self.assertGreater(len(detections), 0)
        for detection in detections:
            self.assertIsInstance(detection, tuple)
            self.assertEqual(len(detection), 2)
            distance, angle = detection
            self.assertIsInstance(distance, (int, float))
            self.assertIsInstance(angle, (int, float))

class TestDQNAgent(unittest.TestCase):
    def setUp(self):
        state_dim = 100
        action_dim = 4
        self.agent = DQNAgent(state_dim, action_dim)

    def test_agent_initialization(self):
        self.assertEqual(self.agent.state_dim, 100)
        self.assertEqual(self.agent.action_dim, 4)
        self.assertIsInstance(self.agent.model, torch.nn.Module)
        self.assertIsInstance(self.agent.target_model, torch.nn.Module)

    def test_get_action(self):
        state = np.random.rand(100)
        action = self.agent.get_action(state)
        self.assertIsInstance(action, int)
        self.assertTrue(0 <= action < 4)

    def test_remember(self):
        initial_memory_size = len(self.agent.memory)
        self.agent.remember(np.zeros(100), 0, 1, np.ones(100), False)
        self.assertEqual(len(self.agent.memory), initial_memory_size + 1)

    def test_train(self):
        # Fill memory with some random data
        for _ in range(32):
            state = np.random.rand(100)
            next_state = np.random.rand(100)
            self.agent.remember(state, 0, 1, next_state, False)
        
        # Train the agent
        self.agent.train()
        # Check if epsilon decayed
        self.assertLess(self.agent.epsilon, 1.0)

if __name__ == '__main__':
    unittest.main()
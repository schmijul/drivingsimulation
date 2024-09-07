import pygame
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from car import Car
from obstacle import Obstacle
from radar import Radar
from slam_map import SlamMap
from dqn_agent import DQNAgent
class Simulator:
    def __init__(self, width, height, car_speed, car_acceleration, car_deceleration, 
                 radar_range, radar_angle, num_obstacles):
        self.width = width * 3  # Triple the width to accommodate all views
        self.height = height
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Car Simulator with Radar, SLAM, and DQN")
        
        self.car = Car(width // 2, height // 2, car_speed, car_acceleration, car_deceleration)
        self.radar = Radar(radar_range, radar_angle)
        self.obstacles = []
        self.clock = pygame.time.Clock()
        self.slam_map = SlamMap(width, height)
        
        self.goal = self.spawn_goal()
        self.spawn_initial_obstacles(num_obstacles)

        self.font = pygame.font.Font(None, 24)

        # DQN Agent
        state_dim = self.slam_map.grid.size + 6  # SLAM map + car position, angle, speed, goal position
        action_dim = 4  # Up, Down, Left, Right
        self.agent = DQNAgent(state_dim, action_dim)

    def spawn_goal(self):
        while True:
            x = random.randint(50, self.width // 3 - 50)
            y = random.randint(50, self.height - 50)
            goal_rect = pygame.Rect(x, y, 20, 20)
            if not any(goal_rect.colliderect(obs.rect) for obs in self.obstacles):
                return goal_rect

    def spawn_initial_obstacles(self, num_obstacles):
        for _ in range(num_obstacles):
            while True:
                size = random.randint(20, 60)
                x = random.randint(0, self.width // 3 - size)  # Limit to left third
                y = random.randint(0, self.height - size)
                new_obstacle = Obstacle(x, y, size, size)
                
                if not self.car.rect.colliderect(new_obstacle.rect) and \
                   not any(new_obstacle.rect.colliderect(obs.rect) for obs in self.obstacles) and \
                   not new_obstacle.rect.colliderect(self.goal):
                    self.obstacles.append(new_obstacle)
                    break

    def check_collision(self):
        return any(self.car.rect.colliderect(obstacle.rect) for obstacle in self.obstacles)

    def check_goal_reached(self):
        return self.car.rect.colliderect(self.goal)

    def draw_radar_view(self, radar_detections):
        radar_surface = pygame.Surface((self.width // 3, self.height))
        radar_surface.fill((0, 0, 0))  # Black background
        
        # Draw a circle representing the maximum radar range
        pygame.draw.circle(radar_surface, (0, 100, 0), (self.width // 6, self.height // 2), self.radar.range, 1)
        
        # Draw the radar cone
        start_angle = -self.radar.angle // 2
        end_angle = self.radar.angle // 2
        pygame.draw.arc(radar_surface, (0, 200, 0), 
                        (self.width // 6 - self.radar.range, self.height // 2 - self.radar.range, 
                         self.radar.range * 2, self.radar.range * 2), 
                        math.radians(start_angle), math.radians(end_angle), 1)

        # Draw detected obstacles
        for detection in radar_detections:
            distance, angle = detection
            x = self.width // 6 + int(distance * math.cos(math.radians(angle)))
            y = self.height // 2 - int(distance * math.sin(math.radians(angle)))
            pygame.draw.circle(radar_surface, (255, 0, 0), (x, y), 5)

        return radar_surface

    def draw_data(self):
        data_surface = pygame.Surface((self.width, 100))
        data_surface.fill((200, 200, 200))  # Light gray background
        
        # Car data
        car_pos = f"Car Position: ({self.car.rect.centerx:.2f}, {self.car.rect.centery:.2f})"
        car_speed = f"Car Speed: {self.car.speed:.2f}"
        car_angle = f"Car Angle: {self.car.angle:.2f}"
        
        # Goal data
        goal_pos = f"Goal Position: ({self.goal.centerx}, {self.goal.centery})"
        
        # Render text
        texts = [
            self.font.render(car_pos, True, (0, 0, 0)),
            self.font.render(car_speed, True, (0, 0, 0)),
            self.font.render(car_angle, True, (0, 0, 0)),
            self.font.render(goal_pos, True, (0, 0, 0))
        ]
        
        # Display text
        for i, text in enumerate(texts):
            data_surface.blit(text, (10, 10 + i * 25))
        
        return data_surface

    def get_state(self):
        slam_state = self.slam_map.grid.flatten()
        car_state = np.array([
            self.car.rect.centerx / self.width,
            self.car.rect.centery / self.height,
            self.car.angle / 360,
            self.car.speed / self.car.max_speed,
            self.goal.centerx / self.width,
            self.goal.centery / self.height
        ])
        return np.concatenate((slam_state, car_state))

    def step(self, action):
        # Convert action to key presses
        keys = {pygame.K_UP: False, pygame.K_DOWN: False, pygame.K_LEFT: False, pygame.K_RIGHT: False}
        if action == 0:
            keys[pygame.K_UP] = True
        elif action == 1:
            keys[pygame.K_DOWN] = True
        elif action == 2:
            keys[pygame.K_LEFT] = True
        elif action == 3:
            keys[pygame.K_RIGHT] = True

        self.car.move(keys)

        # Ensure car stays in the left third
        self.car.rect.clamp_ip(pygame.Rect(0, 0, self.width // 3, self.height))

        collision = self.check_collision()
        goal_reached = self.check_goal_reached()

        reward = -1  # Small negative reward for each step
        done = False

        if collision:
            reward = -100  # Large negative reward for collision
            done = True
            self.slam_map.reset()  # Reset SLAM map after collision
        elif goal_reached:
            reward = 100  # Large positive reward for reaching the goal
            done = True
            self.slam_map.reset()  # Reset SLAM map after reaching goal

        # Update SLAM map
        radar_detections = self.radar.scan(self.car, self.obstacles)
        self.slam_map.update(self.car.rect.center, self.car.angle, radar_detections)

        return self.get_state(), reward, done

    def update_display(self):
        self.screen.fill((255, 255, 255))
        for obstacle in self.obstacles:
            obstacle.draw(self.screen)
        self.radar.draw(self.screen, self.car, self.radar.scan(self.car, self.obstacles))
        self.car.draw(self.screen)
        
        # Draw goal
        pygame.draw.rect(self.screen, (0, 255, 0), self.goal)

        # Radar view (middle third)
        radar_surface = self.draw_radar_view(self.radar.scan(self.car, self.obstacles))
        self.screen.blit(radar_surface, (self.width // 3, 0))

        # SLAM map view (right third)
        slam_surface = self.slam_map.draw()
        self.screen.blit(slam_surface, (2 * self.width // 3, 0))

        # Draw data at the bottom of the screen
        data_surface = self.draw_data()
        self.screen.blit(data_surface, (0, self.height - 100))

        pygame.display.flip()
        self.clock.tick(60)

    def run_manual(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            keys = pygame.key.get_pressed()
            self.car.move(keys)

            # Ensure car stays in the left third
            self.car.rect.clamp_ip(pygame.Rect(0, 0, self.width // 3, self.height))

            if self.check_collision():
                print("Collision detected!")
                self.car.rect.center = (self.width // 6, self.height // 2)
                self.car.angle = 0
                self.car.speed = 0
                self.slam_map.reset()  # Reset SLAM map after collision

            if self.check_goal_reached():
                print("Goal reached!")
                self.goal = self.spawn_goal()
                self.slam_map.reset()  # Reset SLAM map after reaching goal

            # Update SLAM map
            radar_detections = self.radar.scan(self.car, self.obstacles)
            self.slam_map.update(self.car.rect.center, self.car.angle, radar_detections)

            self.update_display()

        pygame.quit()

    def train(self, num_episodes):
        for episode in range(num_episodes):
            state = self.get_state()
            total_reward = 0
            steps = 0
            done = False

            while not done:
                action = self.agent.get_action(state)
                next_state, reward, done = self.step(action)
                self.agent.remember(state, action, reward, next_state, done)
                self.agent.train()

                state = next_state
                total_reward += reward
                steps += 1

                self.update_display()

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return

                if steps >= 1000:  # Max steps per episode
                    done = True

            print(f"Episode: {episode + 1}, Steps: {steps}, Total Reward: {total_reward}")

            if episode % 10 == 0:
                self.agent.update_target_model()

            # Reset the environment
            self.car.rect.center = (self.width // 6, self.height // 2)
            self.car.angle = 0
            self.car.speed = 0
            self.goal = self.spawn_goal()
            self.slam_map.reset()

    def run_trained_agent(self):
        state = self.get_state()
        done = False

        while not done:
            action = self.agent.get_action(state)
            state, _, done = self.step(action)
            self.update_display()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
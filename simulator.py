import pygame
import random
import math
from car import Car
from obstacle import Obstacle
from radar import Radar
from slam_map import SlamMap

class Simulator:
    def __init__(self, width, height, car_speed, car_acceleration, car_deceleration, 
                 radar_range, radar_angle, num_obstacles):
        self.width = width * 3  # Triple the width to accommodate all views
        self.height = height
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Car Simulator with Radar and SLAM")
        
        self.car = Car(width // 2, height // 2, car_speed, car_acceleration, car_deceleration)
        self.radar = Radar(radar_range, radar_angle)
        self.obstacles = []
        self.clock = pygame.time.Clock()
        self.slam_map = SlamMap(width, height)
        
        self.goal = self.spawn_goal()
        self.spawn_initial_obstacles(num_obstacles)

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

    def run(self):
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
                running = False

            if self.check_goal_reached():
                print("Goal reached!")
                self.goal = self.spawn_goal()  # Spawn a new goal

            radar_detections = self.radar.scan(self.car, self.obstacles)

            # Update SLAM map
            self.slam_map.update(self.car.rect.center, self.car.angle, radar_detections)

            # Main simulation view (left third)
            self.screen.fill((255, 255, 255))
            for obstacle in self.obstacles:
                obstacle.draw(self.screen)
            self.radar.draw(self.screen, self.car, radar_detections)
            self.car.draw(self.screen)
            
            # Draw goal
            pygame.draw.rect(self.screen, (0, 255, 0), self.goal)

            # Radar view (middle third)
            radar_surface = self.draw_radar_view(radar_detections)
            self.screen.blit(radar_surface, (self.width // 3, 0))

            # SLAM map view (right third)
            slam_surface = self.slam_map.draw()
            self.screen.blit(slam_surface, (2 * self.width // 3, 0))

            pygame.display.flip()
            self.clock.tick(60)

if __name__ == "__main__":
    pygame.init()
    sim = Simulator(800, 600, 5, 0.1, 0.05, 200, 60, 5)
    sim.run()
    pygame.quit()
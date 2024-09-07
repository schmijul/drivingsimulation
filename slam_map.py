import pygame
import numpy as np
import math

class SlamMap:
    def __init__(self, width, height, resolution=5):
        self.width = width
        self.height = height
        self.resolution = resolution
        self.grid_width = width // resolution
        self.grid_height = height // resolution
        self.grid = np.zeros((self.grid_height, self.grid_width))
        self.surface = pygame.Surface((width, height))
        self.surface.fill((255, 255, 255))  # White background

    def update(self, car_pos, car_angle, radar_detections):
        car_x, car_y = car_pos
        for distance, angle in radar_detections:
            # Calculate absolute angle
            abs_angle = car_angle + angle
            # Calculate obstacle position
            obs_x = car_x + distance * math.cos(math.radians(abs_angle))
            obs_y = car_y - distance * math.sin(math.radians(abs_angle))
            # Convert to grid coordinates
            grid_x = int(obs_x / self.resolution)
            grid_y = int(obs_y / self.resolution)
            if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height:
                self.grid[grid_y, grid_x] = min(self.grid[grid_y, grid_x] + 0.2, 1)

    def draw(self):
        self.surface.fill((255, 255, 255))
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                if self.grid[y, x] > 0:
                    color = (int(255 * (1 - self.grid[y, x])),) * 3
                    pygame.draw.rect(self.surface, color, 
                                     (x * self.resolution, y * self.resolution, 
                                      self.resolution, self.resolution))
        return self.surface
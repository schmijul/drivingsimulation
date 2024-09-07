import pygame
import math

class Car:
    def __init__(self, x, y):
        self.image = pygame.Surface((40, 20), pygame.SRCALPHA)
        pygame.draw.rect(self.image, (255, 0, 0), [0, 0, 40, 20])
        self.rect = self.image.get_rect(center=(x, y))
        self.angle = 0
        self.speed = 0
        self.max_speed = 5
        self.acceleration = 0.1
        self.deceleration = 0.05

    def move(self, keys):
        if keys[pygame.K_UP]:
            self.speed = min(self.speed + self.acceleration, self.max_speed)
        elif keys[pygame.K_DOWN]:
            self.speed = max(self.speed - self.acceleration, -self.max_speed / 2)
        else:
            self.speed *= (1 - self.deceleration)

        if keys[pygame.K_LEFT]:
            self.angle += 2
        if keys[pygame.K_RIGHT]:
            self.angle -= 2

        angle_rad = math.radians(self.angle)
        self.rect.x += self.speed * math.cos(angle_rad)
        self.rect.y -= self.speed * math.sin(angle_rad)

    def draw(self, screen):
        rotated = pygame.transform.rotate(self.image, self.angle)
        screen.blit(rotated, rotated.get_rect(center=self.rect.center))
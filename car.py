import pygame
import math

class Car:
    def __init__(self, x, y, max_speed, acceleration, deceleration):
        self.image = pygame.Surface((40, 20), pygame.SRCALPHA)
        pygame.draw.rect(self.image, (255, 0, 0), [0, 0, 40, 20])
        self.rect = self.image.get_rect(center=(x, y))
        self.angle = 0
        self.speed = 0
        self.max_speed = max_speed
        self.acceleration = acceleration
        self.deceleration = deceleration

    def move(self, keys):
        if keys[pygame.K_UP]:
            self.accelerate()
        elif keys[pygame.K_DOWN]:
            self.decelerate()
        else:
            self.coast()

        if keys[pygame.K_LEFT]:
            self.rotate_left()
        if keys[pygame.K_RIGHT]:
            self.rotate_right()

        self.update_position()

    def accelerate(self):
        self.speed = min(self.speed + self.acceleration, self.max_speed)

    def decelerate(self):
        self.speed = max(self.speed - self.acceleration, -self.max_speed / 2)

    def coast(self):
        self.speed *= (1 - self.deceleration)

    def rotate_left(self):
        self.angle += 2

    def rotate_right(self):
        self.angle -= 2

    def update_position(self):
        angle_rad = math.radians(self.angle)
        self.rect.x += self.speed * math.cos(angle_rad)
        self.rect.y -= self.speed * math.sin(angle_rad)

    def draw(self, screen):
        rotated = pygame.transform.rotate(self.image, self.angle)
        screen.blit(rotated, rotated.get_rect(center=self.rect.center))
    def check_goal_reached(self):
        return self.car.rect.colliderect(self.goal)
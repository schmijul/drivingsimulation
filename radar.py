import pygame
import math
from utils import line_rectangle_intersection

class Radar:
    def __init__(self, range, angle):
        self.range = range
        self.angle = angle

    def scan(self, car, obstacles):
        detections = []
        car_center = car.rect.center
        start_angle = car.angle - self.angle / 2
        end_angle = car.angle + self.angle / 2

        for angle in range(int(start_angle), int(end_angle) + 1, 5):  # 5 degree steps
            angle_rad = math.radians(angle)
            end_x = car_center[0] + self.range * math.cos(angle_rad)
            end_y = car_center[1] - self.range * math.sin(angle_rad)
            
            for obstacle in obstacles:
                intersection = line_rectangle_intersection(car_center, (end_x, end_y), obstacle.rect)
                if intersection:
                    distance = math.sqrt((intersection[0] - car_center[0])**2 + (intersection[1] - car_center[1])**2)
                    detections.append((distance, angle - car.angle))
                    break

        return detections

    def draw(self, screen, car, detections):
        car_center = car.rect.center
        start_angle = car.angle - self.angle / 2
        end_angle = car.angle + self.angle / 2

        for angle in range(int(start_angle), int(end_angle) + 1, 5):
            angle_rad = math.radians(angle)
            end_x = car_center[0] + self.range * math.cos(angle_rad)
            end_y = car_center[1] - self.range * math.sin(angle_rad)
            pygame.draw.line(screen, (0, 255, 0), car_center, (end_x, end_y), 1)

        for detection in detections:
            distance, rel_angle = detection
            angle_rad = math.radians(car.angle + rel_angle)
            end_x = car_center[0] + distance * math.cos(angle_rad)
            end_y = car_center[1] - distance * math.sin(angle_rad)
            pygame.draw.circle(screen, (255, 0, 0), (int(end_x), int(end_y)), 5)
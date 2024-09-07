import pygame
from simulator import Simulator

def main():
    # Simulation parameters
    width = 800
    height = 600
    car_speed = 5
    car_acceleration = 0.1
    car_deceleration = 0.05
    radar_range = 200
    radar_angle = 60
    num_obstacles = 5

    pygame.init()
    simulator = Simulator(width, height, car_speed, car_acceleration, car_deceleration, 
                          radar_range, radar_angle, num_obstacles)
    simulator.run()
    pygame.quit()

if __name__ == "__main__":
    main()
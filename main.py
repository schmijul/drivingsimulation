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
    mode = "manual"# ai or manual
    num_episodes = 10
    pygame.init()
    simulator = Simulator(width, height, car_speed, car_acceleration, car_deceleration, 
                          radar_range, radar_angle, num_obstacles)

 
    if mode == "manual":
        simulator.run_manual()
    elif mode == "ai":
       
        simulator.train(num_episodes)
        simulator.run_trained_agent()
    else:
        print("Invalid mode selected. Exiting.")

    pygame.quit()

if __name__ == "__main__":
    main()
import pygame
from simulator import Simulator

def main():
    pygame.init()
    simulator = Simulator(800, 600)
    simulator.run()
    pygame.quit()

if __name__ == "__main__":
    main()
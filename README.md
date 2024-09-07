# Car Simulator with SLAM

This project is a learning exercise in creating a 2D car simulator with radar and Simultaneous Localization and Mapping (SLAM) functionality using Python and Pygame.

## Project Overview

This simulator features a car that can be controlled by the user, moving around in an environment with randomly placed obstacles. The car is equipped with a radar that detects these obstacles. As the car moves, it builds a map of its environment using a basic SLAM algorithm.

The simulation window is divided into three parts:
1. The main simulation view (left)
2. The radar view (center)
3. The SLAM map view (right)

This project serves as an introduction to concepts such as:
- Basic game development with Pygame
- Simulating sensors (radar)
- Rudimentary SLAM implementation
- Object-oriented programming in Python
- Unit testing

## Files and Their Functions

1. `main.py`: The entry point of the program. It initializes the simulator with configurable parameters.

2. `simulator.py`: Contains the `Simulator` class, which manages the overall simulation, including the car, obstacles, radar, and SLAM map.

3. `car.py`: Defines the `Car` class, representing the vehicle that the user can control.

4. `obstacle.py`: Contains the `Obstacle` class, used to create the objects that the car must avoid.

5. `radar.py`: Implements the `Radar` class, simulating a simple radar system for obstacle detection.

6. `slam_map.py`: Defines the `SlamMap` class, which creates and updates the map based on radar data.

7. `test_simulator.py`: Contains unit tests for various components of the simulator.

## How the Mapping Works

The SLAM (Simultaneous Localization and Mapping) functionality in this project is a simplified version of real-world SLAM algorithms. Here's how it works:

1. The environment is divided into a grid. Each cell in the grid represents a small area of the environment.

2. As the car moves, the radar detects obstacles and their positions relative to the car.

3. The SLAM map takes this radar data and updates the corresponding cells in the grid.

4. Each time an obstacle is detected in a cell, the cell's value is increased (up to a maximum of 1).

5. The map is visualized by coloring each cell based on its value. Darker cells represent areas where obstacles are more likely to be present.

This approach allows the map to be built up over time as the car explores the environment. It's a basic implementation and doesn't account for factors like sensor noise or localization errors, which more advanced SLAM algorithms would handle.

## Installation

1. Ensure you have Python 3.x installed on your system.

2. Install the required library:
   ```
   pip install pygame
   ```

3. Clone this repository or download the source files.

## Running the Simulator

1. Navigate to the project directory in your terminal.

2. Run the following command:
   ```
   python main.py
   ```

3. Use the arrow keys to control the car:
   - Up: Accelerate
   - Down: Decelerate/Reverse
   - Left/Right: Turn

## Running the Tests

To run the unit tests, use the following command in the project directory:

```
python -m unittest test_simulator.py
`
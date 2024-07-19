using SFML.Graphics;
using SFML.Window;
using SFML.System;
using System.Collections.Generic;

namespace SelfDrivingSimulation
{
    public class Simulation
    {
        private RenderWindow window;
        private Vehicle vehicle;
        private VehicleController controller;
        private List<Obstacle> obstacles;

        public Simulation(Vehicle vehicle, VehicleController controller)
        {
            this.vehicle = vehicle;
            this.controller = controller;
            window = new RenderWindow(new VideoMode(800, 600), "2D Self-Driving Simulation");
            window.Closed += (sender, e) => window.Close();

            obstacles = new List<Obstacle>
            {
                new Obstacle(200, 200, 20),
                new Obstacle(600, 400, 30),
                new Obstacle(400, 500, 25)
            };
        }

        public void Run()
        {
            while (window.IsOpen)
            {
                window.DispatchEvents();

                HandleInput();
                controller.UpdateVehicle();
                vehicle.Move();

                if (CheckCollision())
                {
                    controller.ReportCollision();
                }

                window.Clear(Color.White);
                vehicle.Draw(window);
                foreach (var obstacle in obstacles)
                {
                    obstacle.Draw(window);
                }
                window.Display();
            }
        }

        private bool CheckCollision()
        {
            foreach (var obstacle in obstacles)
            {
                if (vehicle.CollidesWith(obstacle))
                {
                    return true;
                }
            }
            return false;
        }

        private void HandleInput()
        {
            if (Keyboard.IsKeyPressed(Keyboard.Key.Up))
                controller.ProcessCommand("accelerate", 0.1f);
            else if (Keyboard.IsKeyPressed(Keyboard.Key.Down))
                controller.ProcessCommand("accelerate", -0.1f);
            else
                controller.ProcessCommand("accelerate", 0f);

            if (Keyboard.IsKeyPressed(Keyboard.Key.Left))
                controller.ProcessCommand("turn", -2f);
            if (Keyboard.IsKeyPressed(Keyboard.Key.Right))
                controller.ProcessCommand("turn", 2f);
        }
    }
}
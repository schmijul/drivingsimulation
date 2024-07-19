using System;

namespace SelfDrivingSimulation
{
    public class VehicleController
    {
        private Vehicle vehicle;
        private string lastCommand = "";
        private float lastValue = 0f;
        private bool collided = false;

        public VehicleController(Vehicle vehicle)
        {
            this.vehicle = vehicle;
        }

        public void ProcessCommand(string command, float value)
        {
            lastCommand = command.ToLower();
            lastValue = value;
        }

        public void UpdateVehicle()
        {
            switch (lastCommand)
            {
                case "accelerate":
                    vehicle.Accelerate(lastValue);
                    break;
                case "turn":
                    vehicle.Turn(lastValue);
                    break;
                case "setspeed":
                    vehicle.Speed = Math.Clamp(lastValue, -vehicle.MaxSpeed, vehicle.MaxSpeed);
                    break;
                case "setangle":
                    vehicle.Angle = lastValue % 360;
                    if (vehicle.Angle < 0) vehicle.Angle += 360;
                    break;
            }
        }

        public void ReportCollision()
        {
            if (!collided)
            {
                collided = true;
                Console.WriteLine("Collision detected! The vehicle has hit an obstacle.");
            }
        }

        public void ResetCollision()
        {
            collided = false;
        }

        public bool HasCollided()
        {
            return collided;
        }
    }
}
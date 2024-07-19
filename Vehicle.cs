using SFML.Graphics;
using SFML.System;
using System;

namespace SelfDrivingSimulation
{
    public class Vehicle
    {
        public Vector2f Position { get; set; }
        public float Angle { get; set; }
        public float Speed { get; set; }
        public float MaxSpeed { get; set; } = 5f;
        public float Length { get; set; } = 40f;
        public float Width { get; set; } = 20f;
        public float FieldOfViewAngle { get; set; } = 60f;
        public float FieldOfViewDistance { get; set; } = 100f;

        private ConvexShape carShape;
        private ConvexShape fovShape;

        public Vehicle()
        {
            Position = new Vector2f(400, 300); // Default starting position
            Angle = 0;
            Speed = 0;

            carShape = new ConvexShape(4);
            UpdateCarShape();

            fovShape = new ConvexShape(3);
            UpdateFOVShape();
        }

        private void UpdateCarShape()
        {
            float halfLength = Length / 2;
            float halfWidth = Width / 2;

            carShape.SetPoint(0, new Vector2f(-halfLength, -halfWidth));
            carShape.SetPoint(1, new Vector2f(halfLength, -halfWidth));
            carShape.SetPoint(2, new Vector2f(halfLength, halfWidth));
            carShape.SetPoint(3, new Vector2f(-halfLength, halfWidth));

            carShape.FillColor = Color.Red;
            carShape.OutlineColor = Color.Black;
            carShape.OutlineThickness = 2;
        }

        private void UpdateFOVShape()
        {
            fovShape.SetPoint(0, new Vector2f(Length / 2, 0));
            fovShape.SetPoint(1, new Vector2f(
                FieldOfViewDistance * (float)Math.Cos(-FieldOfViewAngle * Math.PI / 360),
                FieldOfViewDistance * (float)Math.Sin(-FieldOfViewAngle * Math.PI / 360)));
            fovShape.SetPoint(2, new Vector2f(
                FieldOfViewDistance * (float)Math.Cos(FieldOfViewAngle * Math.PI / 360),
                FieldOfViewDistance * (float)Math.Sin(FieldOfViewAngle * Math.PI / 360)));

            fovShape.FillColor = new Color(0, 255, 0, 100);
        }

        public void Accelerate(float amount)
        {
            Speed = Math.Clamp(Speed + amount, -MaxSpeed, MaxSpeed);
        }

        public void Decelerate(float amount)
        {
            Speed = Math.Clamp(Speed - amount, 0, MaxSpeed);
        }

        public void Turn(float degrees)
        {
            Angle += degrees;
            Angle %= 360;
            if (Angle < 0) Angle += 360;
        }

        public void Move()
        {
            float radians = Angle * (float)Math.PI / 180;
            Position += new Vector2f(
                Speed * (float)Math.Cos(radians),
                Speed * (float)Math.Sin(radians)
            );
        }

        public void Draw(RenderWindow window)
        {
            carShape.Position = Position;
            carShape.Rotation = Angle;
            window.Draw(carShape);

            fovShape.Position = Position;
            fovShape.Rotation = Angle;
            window.Draw(fovShape);
        }

        public bool CollidesWith(Obstacle obstacle)
        {
            float distance = Vector2f.Distance(Position, obstacle.Position);
            return distance < (Length / 2 + obstacle.Radius);
        }
    }
}
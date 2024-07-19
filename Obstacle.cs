using SFML.Graphics;
using SFML.System;

namespace SelfDrivingSimulation
{
    public class Obstacle
    {
        public Vector2f Position { get; set; }
        public float Radius { get; set; }
        private CircleShape shape;

        public Obstacle(float x, float y, float radius)
        {
            Position = new Vector2f(x, y);
            Radius = radius;
            shape = new CircleShape(radius);
            shape.FillColor = Color.Blue;
            shape.Position = new Vector2f(x - radius, y - radius);
        }

        public void Draw(RenderWindow window)
        {
            window.Draw(shape);
        }
    }
}
using SFML.Graphics;
using SFML.Window;
using SFML.System;

class Program
{
    static void Main(string[] args)
    {
        var window = new RenderWindow(new VideoMode(800, 600), "2D Self-Driving Simulation");
        window.Closed += (sender, e) => window.Close();

        var vehicle = new Vehicle(400, 300);

        while (window.IsOpen)
        {
            window.DispatchEvents();

            // Handle input
            if (Keyboard.IsKeyPressed(Keyboard.Key.Up))
                vehicle.Speed = 2;
            else if (Keyboard.IsKeyPressed(Keyboard.Key.Down))
                vehicle.Speed = -2;
            else
                vehicle.Speed = 0;

            if (Keyboard.IsKeyPressed(Keyboard.Key.Left))
                vehicle.Angle += 2;
            if (Keyboard.IsKeyPressed(Keyboard.Key.Right))
                vehicle.Angle -= 2;

            vehicle.Move();

            window.Clear(Color.White);
            vehicle.Draw(window);
            window.Display();
        }
    }
}

class Vehicle
{
    public float X { get; set; }
    public float Y { get; set; }
    public float Angle { get; set; }
    public float Speed { get; set; }

    private CircleShape shape;

    public Vehicle(float x, float y)
    {
        X = x;
        Y = y;
        Angle = 0;
        Speed = 0;

        shape = new CircleShape(10)
        {
            FillColor = Color.Red
        };
    }

    public void Move()
    {
        X += Speed * (float)Math.Cos(Angle * Math.PI / 180);
        Y -= Speed * (float)Math.Sin(Angle * Math.PI / 180);
    }

    public void Draw(RenderWindow window)
    {
        shape.Position = new Vector2f(X - shape.Radius, Y - shape.Radius);
        window.Draw(shape);
    }
}
using Microsoft.AspNetCore.Builder;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using System.Threading.Tasks;
using SFML.Graphics;
using SFML.Window;
using SFML.System;

namespace SelfDrivingSimulation
{
    class Program
    {
        static async Task Main(string[] args)
        {
            var builder = WebApplication.CreateBuilder(args);

            // Add services to the container.
            builder.Services.AddControllers();
            builder.Services.AddSingleton<Vehicle>();
            builder.Services.AddSingleton<VehicleController>();
            builder.Services.AddSingleton<Simulation>();

            var app = builder.Build();

            // Configure the HTTP request pipeline.
            app.UseHttpsRedirection();
            app.UseAuthorization();
            app.MapControllers();

            // Run the API server
            _ = app.RunAsync();

            // Run the simulation
            var simulation = app.Services.GetRequiredService<Simulation>();
            simulation.Run();
        }
    }
}
using Microsoft.AspNetCore.Mvc;

namespace SelfDrivingSimulation
{
    [ApiController]
    [Route("api/vehicle")]
    public class VehicleApiController : ControllerBase
    {
        private readonly VehicleController _vehicleController;

        public VehicleApiController(VehicleController vehicleController)
        {
            _vehicleController = vehicleController;
        }

        [HttpPost("command")]
        public IActionResult SendCommand([FromBody] CommandModel command)
        {
            _vehicleController.ProcessCommand(command.Command, command.Value);
            return Ok(new { message = $"Command '{command.Command}' processed with value {command.Value}" });
        }

        [HttpGet("collision")]
        public IActionResult GetCollisionStatus()
        {
            return Ok(new { collision = _vehicleController.HasCollided() });
        }
    }

    public class CommandModel
    {
        public string Command { get; set; }
        public float Value { get; set; }
    }
}
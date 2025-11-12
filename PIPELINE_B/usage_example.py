"""
Example usage of the VoiceCommandController library
"""

from voice_command_controller import VoiceCommandController


# Example 1: Simple usage with callback
def command_handler(service, params, execute):
    """Handle executed commands"""
    if execute:
        print(f"✅ Executing: {service} with params {params}")
        # Add your ROS service call or other logic here
        # For example:
        # ros_client.call_service(service, params)
    else:
        print(f"❌ Command cancelled: {service}")


# Create controller and run
controller = VoiceCommandController(
    audio_device_index=2,  # Change as needed
    wake_word_path="models/Drone-Swarm_en_windows_v3_0_0.ppn",
    vosk_model_path="models/vosk-model-small-en-us-0.15"
)

controller.set_command_callback(command_handler)
controller.run()


# Example 2: Using context manager
def my_command_callback(service, params, execute):
    if execute:
        print(f"Service: {service}, Params: {params}")


with VoiceCommandController() as controller:
    controller.set_command_callback(my_command_callback)
    controller.run()

# Example 3: Manual control (non-blocking)
controller = VoiceCommandController()

try:
    while True:
        # Check for wake word
        if controller.process_wake_word_detection():
            print("Wake word detected!")

            # Handle the command
            result = controller.handle_voice_command()

            if result:
                service, params, execute = result
                if execute:
                    print(f"Execute: {service}({params})")
                    # Your custom logic here

except KeyboardInterrupt:
    print("\nStopping...")
finally:
    controller.cleanup()

# Example 4: Get list of supported commands
commands = VoiceCommandController.get_supported_commands()
print("Supported commands:")
for cmd in commands:
    print(f"  - {cmd}")

# Example 5: Parse command without full pipeline
controller = VoiceCommandController()
service, params = controller.parse_command("start scan one")
print(f"Command parsed: {service} with {params}")
controller.cleanup()
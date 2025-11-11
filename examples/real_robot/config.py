"""Configuration for InternVLA-M1 Real Robot Deployment."""

from dataclasses import dataclass


@dataclass
class deploy_config:
    """Deployment configuration for real robot control."""
    
    # Language instruction
    language_instruction: str = "Make a classic burger set."
    
    # Communication
    inference_server_url: str = "http://0.0.0.0:25553/infer"
    realsense_server_url: str = "http://0.0.0.0:5021"
    
    # Logging
    log_action_to_json: bool = True
    
    # Action configuration
    action_type: str = "delta_ee_pose"
    fps: int = 10
    
    # Robot configuration
    franka_hostname: str = "172.16.0.2"
    gripper_type: str = "robotiq"
    gripper_port: str = "/dev/ttyUSB0"
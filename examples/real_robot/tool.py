"""
Utility functions for robot deployment.

This module contains helper functions for camera data processing,
pose transformations, and data formatting.
"""

import base64

import cv2
import numpy as np
import requests
from scipy.spatial.transform import Rotation as R


def request_image(address="0.0.0.0", port=5021):
    """
    Request image data from camera server.
    
    Args:
        address: Camera server IP address
        port: Camera server port
        
    Returns:
        JSON response containing image data
    """
    url = f"http://{address}:{port}/image"
    headers = {"Content-Type": "application/json"}
    response = requests.get(url, headers=headers)
    return response.json()


def get_camera_data(address="0.0.0.0", port="5021"):
    """
    Get camera data from dual camera setup.
    
    Args:
        address: Camera server IP address
        port: Camera server port
        
    Returns:
        Tuple of (colors, depths) where each is a list of 2 camera views
    """
    images = request_image(address=address, port=port)
    
    # Process data from two cameras
    colors_0 = base64.b64decode(images["data"]["colors"][0])
    colors_0 = cv2.imdecode(np.frombuffer(colors_0, np.uint8), cv2.IMREAD_COLOR)
    
    colors_1 = base64.b64decode(images["data"]["colors"][1])
    colors_1 = cv2.imdecode(np.frombuffer(colors_1, np.uint8), cv2.IMREAD_COLOR)
    
    depth_0 = np.frombuffer(
        base64.b64decode(images["data"]["depths"][0]), dtype=np.float64
    ).reshape(480, 640)
    
    depth_1 = np.frombuffer(
        base64.b64decode(images["data"]["depths"][1]), dtype=np.float64
    ).reshape(480, 640)
    
    return [colors_0, colors_1], [depth_0, depth_1]


def pose_to_6d(pose, degrees=False):
    """
    Convert 4x4 pose matrix to 6D representation (xyz + rpy).
    
    Args:
        pose: 4x4 transformation matrix
        degrees: Whether to return rotation in degrees (default: False, returns radians)
        
    Returns:
        6D pose array: [x, y, z, roll, pitch, yaw]
    """
    pose6d = np.zeros(6)
    pose6d[:3] = pose[:3, 3]
    pose6d[3:6] = R.from_matrix(pose[:3, :3]).as_euler("xyz", degrees=degrees)
    return pose6d


def _6d_to_pose(
    pose6d,
    degrees=False
):
    pose = np.eye(4)
    pose[:3, 3] = pose6d[:3]
    pose[:3, :3] = R.from_euler("xyz", pose6d[3:6], degrees=degrees).as_matrix()

    return pose

def formulate_input(data):
    """
    Formulate input data for robot state processing.
    
    Args:
        data: Dictionary containing 'current_pose' (4x4 matrix) and 
              'current_gripper_state' (-1 or 1)
              
    Returns:
        7D observation array: [x, y, z, roll, pitch, yaw, gripper]
    """
    state_pose = np.array(data["current_pose"])  # Shape: (4, 4)
    state_xyz = state_pose[:3, 3]  # Extract translation (3,)
    state_rotation_matrices = state_pose[:3, :3]  # Extract rotation matrices (3, 3)
    state_rot = R.from_matrix(state_rotation_matrices)
    state_rpy = state_rot.as_euler('xyz', degrees=False)  # Convert to RPY (3,)
    state_gripper = np.array(data['current_gripper_state'])  # Gripper state: -1 or 1

    obs_abs_ee_pose = np.hstack((state_xyz, state_rpy, [state_gripper]))  # Shape: (7,)
    return obs_abs_ee_pose


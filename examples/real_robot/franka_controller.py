      
from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from frankx import (
    Affine,
    InvalidOperationException,
    JointMotion,
    Robot,
    Waypoint,
    WaypointMotion,
)
from scipy.spatial.transform import Rotation as R, Slerp

# Type aliases for clarity
PoseMatrix = np.ndarray  # 4x4 homogeneous transformation matrix
PoseQuat = List[float]   # [x, y, z, qw, qx, qy, qz] or [x, y, z, qx, qy, qz, qw] depending on scalar_first
Action = Tuple[Union[float, List[float]], Union[PoseMatrix, PoseQuat]]

PI = np.pi
HOME_JOINTS = [0, -PI / 4, 0, -3 * PI / 4, 0, PI / 2, 0]
HOME_POSE = Affine(0.3069, 0.0, 0.4867, 0, 0, 0.0)
HOME_POSE_ARRAY = [0.3069, 0.0, 0.4867, 0, 0, 0.0]


class FrankaController:
    """Franka Emika Panda robot controller supporting pose-based teleoperation and gripper control."""

    def __init__(
        self,
        hostname: str = "172.16.0.2",
        gripper_type: str = "panda_hand",
        control_type: str = "pose",
        gripper_port: str = "/dev/ttyUSB0",
        reset: bool = True,
    ) -> None:
        """Initialize the Franka robot controller.

        Args:
            hostname: IP address of the Franka Control Interface (FCI).
            gripper_type: Type of gripper ('panda_hand' or 'robotiq').
            control_type: Control mode ('pose' only supported currently).
            gripper_port: Serial port for Robotiq gripper (if used).
            reset: Whether to move to home position on initialization.
        """
        self.robot = Robot(hostname)
        self.gripper_type = gripper_type
        self.initial_reset = reset

        if gripper_type == "panda_hand":
            self.gripper = self.robot.get_gripper()
            self.gripper.gripper_speed = 0.2
            self.gripper.gripper_force = 5.0
        elif gripper_type == "robotiq":
            from robotiq import RobotiqCGripper

            self.gripper = RobotiqCGripper(port=gripper_port)
            self.gripper.wait_for_connection()
        else:
            self.gripper = None

        self.control_type = control_type
        self.robot.set_default_behavior()
        self.robot.recover_from_errors()

        self.robot.velocity_rel = 1.0 / 3
        self.robot.acceleration_rel = 0.6 / 3
        self.robot.jerk_rel = 0.01 / 3

        self.gripper_width: float = 0.08
        self.gripper_open: int = 1

        self.last_action: Optional[Action] = None
        self.current_pose: Optional[Any] = None  # frankx.Affine
        print("Finished initializing")

        if self.initial_reset:
            if self.control_type == "pose":
                move_thread = self.robot.move_async(JointMotion(HOME_JOINTS))
                if self.gripper is not None:
                    self.gripper_open = 1
                    if self.gripper_type == "robotiq":
                        self.gripper.open()
                    else:
                        self.gripper.open()
            elif self.control_type == "joint":
                raise NotImplementedError("Joint control is not implemented.")
            move_thread.join()

        current_pose = self.get_obs()
        print("Current pose:", repr(current_pose["current_pose"]))

        self.start()
        print("Finished start thread.")

    def joint_reset(self) -> None:
        """Reset robot to home joint configuration."""
        move_thread = self.robot.move_async(JointMotion(HOME_JOINTS))
        move_thread.join()

    def start(self) -> threading.Thread:
        """Start asynchronous waypoint motion thread for continuous pose control.

        Returns:
            The motion execution thread.
        """
        print("==== Start Control Thread ====")
        if self.control_type == "pose":
            robot_current_pose_quat = self.get_obs()["current_pose_quat"]
            robot_tgt_pose = Affine(*robot_current_pose_quat)
            self.robot_waypoint_motion = WaypointMotion(
                [Waypoint(robot_tgt_pose)], return_when_finished=False
            )
            self.robot_motion_thread = self.robot.move_async(self.robot_waypoint_motion)
            return self.robot_motion_thread
        elif self.control_type == "joint":
            raise NotImplementedError("Joint control is not implemented.")
        else:
            raise ValueError(f"Unsupported control type: {self.control_type}")

    def reset(self) -> None:
        """Reset the robot and gripper to initial state."""
        time.sleep(0.5)
        self.robot_waypoint_motion.finish()
        self.robot_motion_thread.join()
        if self.gripper is not None:
            self.gripper.open()
        self.robot.velocity_rel = 1.0 / 3
        self.robot.acceleration_rel = 0.6 / 3
        self.robot.jerk_rel = 0.01 / 3
        self.gripper_width = 0.08
        self.gripper_open = 1
        self.last_action = None
        self.current_pose = None

        if self.control_type == "pose":
            move_thread = self.robot.move_async(JointMotion(HOME_JOINTS))
            if self.gripper is not None:
                self.gripper_open = 1
                if self.gripper_type == "robotiq":
                    self.gripper.open(block=False)
                else:
                    self.gripper.open()
        elif self.control_type == "joint":
            raise NotImplementedError("Joint control is not implemented.")
        move_thread.join()

        self.start()
        print("Finished reset")
        time.sleep(0.5)

    def end(self) -> None:
        """Gracefully stop motion and return to home position."""
        self.robot_waypoint_motion.finish()
        self.robot_motion_thread.join()
        self.robot.move(JointMotion(HOME_JOINTS))
        if self.gripper is not None:
            if self.gripper_type == "robotiq":
                self.gripper.open(block=False)
            else:
                self.gripper.homing()

    def get_robot_state(self) -> np.ndarray:
        """Get current joint positions and gripper state.

        Args:
            read_gripper: Placeholder; gripper state is always included.

        Returns:
            Concatenated array of joint angles and two identical gripper finger positions.
        """
        try:
            self.current_joint = self.robot.current_joint_positions(read_once=True)
        except InvalidOperationException:
            self.current_joint = self.robot.current_joint_positions(read_once=False)
        state = np.concatenate([self.current_joint, [self.gripper_width / 2, self.gripper_width / 2]])
        return state

    def get_obs(self) -> Dict[str, Any]:
        """Retrieve full observation dictionary from the robot.

        Returns:
            Dictionary containing:
                - current_joint: 7D joint angles
                - panda_hand_pose / current_pose: 4x4 pose matrix
                - current_pose_quat: [x, y, z, qw, qx, qy, qz]
                - current_gripper_width: float
                - state: concatenated state vector
        """
        try:
            self.current_pose = self.robot.current_pose(read_once=True)
        except InvalidOperationException:
            self.current_pose = self.robot.current_pose(read_once=False)

        try:
            self.current_joint = self.robot.current_joint_positions(read_once=True)
        except InvalidOperationException:
            self.current_joint = self.robot.current_joint_positions(read_once=False)

        trans = self.current_pose.translation().tolist()
        rot = self.current_pose.rotation()
        pose = np.eye(4)
        pose[:3, :3] = rot
        pose[:3, 3] = trans

        current_pose_quat = self.current_pose.translation().tolist() + self.current_pose.quaternion()
        obs = {
            "current_joint": self.current_joint,
            "panda_hand_pose": pose,
            "current_pose": pose,
            "current_pose_quat": current_pose_quat,
            "current_gripper_width": self.gripper_width,
            "state": np.concatenate([self.current_joint, [self.gripper_width / 2, self.gripper_width / 2]]),
        }
        return obs

    def apply_action(self, action: Action, type: str = "pose") -> None:
        """Apply a single action to the robot and gripper.

        Args:
            action: Tuple of (gripper_width, end-effector pose).
            type: Action representation ('pose' for 4x4 matrix, 'pose_quat' for [x,y,z,qw,qx,qy,qz]).
        """
        if type == "joint":
            raise NotImplementedError("Joint control is not implemented.")
        elif type == "pose":
            gripper_width, ee_pose = action
            gripper_width = gripper_width if isinstance(gripper_width, float) else sum(gripper_width)
            gripper_width = 0.08 if gripper_width >= 0.04 else 0.0

            trans = ee_pose[:3, 3]
            quat = R.from_matrix(ee_pose[:3, :3]).as_quat(scalar_first=True)
            waypoint_affine = Affine(*np.concatenate([trans, quat]).tolist())
            self.robot_waypoint_motion.set_next_waypoint(Waypoint(waypoint_affine))
        elif type == "pose_quat":
            gripper_width, ee_pose_quat = action
            gripper_width = gripper_width if isinstance(gripper_width, float) else sum(gripper_width)
            gripper_width = 0.08 if gripper_width >= 0.04 else 0.0
            self.robot_waypoint_motion.set_next_waypoint(Waypoint(Affine(*ee_pose_quat)))
        else:
            raise ValueError(f"Unknown action type: {type}")

        gripper_open = 1 if gripper_width > 0.04 else 0

        if gripper_open != self.gripper_open:
            print("gripper change from", self.gripper_open, "to", gripper_open)
            self.gripper_open = gripper_open
            if gripper_open == 1:
                if self.gripper_type == "robotiq":
                    self.gripper.open(block=True)
                else:
                    self.gripper.open()
                self.gripper_width = 0.08
            else:
                if self.gripper_type == "robotiq":
                    self.gripper.close(block=True)
                else:
                    self.gripper.clamp()
                self.gripper_width = 0.0

    def step(self, cur_action: Action, type: str = "pose", interpolate: int = 0) -> Dict[str, Any]:
        """Execute one environment step with optional interpolation.

        Args:
            cur_action: Desired action.
            type: Action type ('pose', 'pose_quat').
            interpolate: Number of intermediate steps for smooth motion (0 = no interpolation).

        Returns:
            Observation dictionary after applying the action.
        """
        if type not in {"joint", "pose", "pose_quat"}:
            raise ValueError(f"Invalid action type: {type}")
        if type == "joint":
            raise NotImplementedError("Joint control is not implemented.")

        if interpolate == 0 or self.last_action is None:
            self.apply_action(cur_action, type=type)
        else:
            num_points = interpolate + 2
            action_list = interpolate_ee_pose(
                last_action=self.last_action,
                cur_action=cur_action,
                num_points=num_points,
                type=type,
            )
            for step in range(1, num_points):
                self.apply_action(action_list[step], type=type)
        self.last_action = cur_action
        return self.get_obs()

    def init_robot(self) -> None:
        """Reinitialize robot to safe home state."""
        self.robot_waypoint_motion.finish()
        self.robot_motion_thread.join()
        self.robot.move(JointMotion(HOME_JOINTS))
        self.robot.recover_from_errors()

        if self.gripper is not None:
            self.gripper_open = 1
            if self.gripper_type == "robotiq":
                self.gripper.open(block=False)
            else:
                self.gripper.open()
        print("Finished init_robot.")


def interpolate_ee_pose(
    last_action: Action,
    cur_action: Action,
    num_points: int,
    type: str,
) -> List[Action]:
    """Interpolate between two end-effector poses using SLERP for rotation and linear for translation.

    Args:
        last_action: Starting action (gripper, pose).
        cur_action: Target action (gripper, pose).
        num_points: Number of interpolated points (including start and end).
        type: Pose representation ('pose' or 'pose_quat').

    Returns:
        List of interpolated actions.
    """
    print("interpolate_ee_pose working once again")
    if type == "joint":
        raise NotImplementedError("Joint interpolation not supported.")
    elif type == "pose":
        start_gripper_width, start_pose = last_action
        end_gripper_width, end_pose = cur_action
        start_trans = start_pose[:3, 3]
        end_trans = end_pose[:3, 3]
        start_rot = R.from_matrix(start_pose[:3, :3])
        end_rot = R.from_matrix(end_pose[:3, :3])
    elif type == "pose_quat":
        start_gripper_width, start_pose_quat = last_action
        end_gripper_width, end_pose_quat = cur_action
        start_trans = start_pose_quat[:3]
        end_trans = end_pose_quat[:3]
        start_rot = R.from_quat(start_pose_quat[3:], scalar_first=True)
        end_rot = R.from_quat(end_pose_quat[3:], scalar_first=True)
    else:
        raise ValueError(f"Unknown action type: {type}")

    interp_grip = [
        start_gripper_width + (end_gripper_width - start_gripper_width) * t / (num_points - 1)
        for t in range(num_points)
    ]
    interp_trans = np.linspace(start_trans, end_trans, num_points)

    key_times = [0, 1]
    interp_times = np.linspace(0, 1, num_points)
    slerp = Slerp(
        key_times,
        R.from_quat(
            [start_rot.as_quat(scalar_first=True), end_rot.as_quat(scalar_first=True)],
            scalar_first=True,
        ),
    )
    interp_rots = slerp(interp_times).as_matrix()

    interpolated_poses: List[Action] = []
    for i in range(num_points):
        if type == "pose":
            interp_pose = np.eye(4)
            interp_pose[:3, 3] = interp_trans[i]
            interp_pose[:3, :3] = interp_rots[i]
            interpolated_poses.append((interp_grip[i], interp_pose))
        elif type == "pose_quat":
            trans = interp_trans[i]
            quat = R.from_matrix(interp_rots[i]).as_quat(scalar_first=True)
            interp_pose = np.concatenate([trans, quat]).tolist()
            interpolated_poses.append((interp_grip[i], interp_pose))
        else:
            raise ValueError(f"Unknown action type during interpolation: {type}")

    return interpolated_poses
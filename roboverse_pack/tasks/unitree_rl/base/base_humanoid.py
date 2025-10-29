from __future__ import annotations

import torch

from roboverse_learn.rl.unitree_rl.helper import get_indices_from_substring

from .base_legged_robot import LeggedRobotTask


class HumanoidTask(LeggedRobotTask):
    """Humanoid specializations on top of LeggedRobotTask."""

    def _init_rigid_body_indices(self):
        robot = self.robot
        sorted_body_names: list[str] = self.sorted_body_names
        self.knee_indices = get_indices_from_substring(robot.knee_links, sorted_body_names).to(self.device)
        self.elbow_indices = get_indices_from_substring(robot.elbow_links, sorted_body_names).to(self.device)
        self.wrist_indices = get_indices_from_substring(robot.wrist_links, sorted_body_names).to(self.device)
        self.torso_indices = get_indices_from_substring(robot.torso_links, sorted_body_names).to(self.device)
        self.ankle_indices = get_indices_from_substring(robot.ankle_links, sorted_body_names).to(self.device)
        return super()._init_rigid_body_indices()

    def _init_joint_cfg(self):
        robot = self.robot
        sorted_joint_names: list[str] = self.sorted_joint_names
        self.left_hip_yaw_roll_joint_indices = get_indices_from_substring(
            robot.left_hip_yaw_roll_joints, sorted_joint_names
        ).to(self.device)
        self.right_hip_yaw_roll_joint_indices = get_indices_from_substring(
            robot.right_hip_yaw_roll_joints, sorted_joint_names
        ).to(self.device)
        self.hip_yaw_roll_joint_indices = torch.concat([
            self.left_hip_yaw_roll_joint_indices,
            self.right_hip_yaw_roll_joint_indices,
        ])
        self.waist_joint_indices = get_indices_from_substring(robot.waist_joints, sorted_joint_names).to(self.device)
        self.arm_joint_indices = get_indices_from_substring(robot.arm_joints, sorted_joint_names).to(self.device)
        return super()._init_joint_cfg()

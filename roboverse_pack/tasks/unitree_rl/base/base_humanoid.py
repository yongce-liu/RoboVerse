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
        return super()._init_rigid_body_indices()

    def _init_joint_cfg(self):
        robot = self.robot
        sorted_joint_names: list[str] = self.sorted_joint_names
        self.left_yaw_roll_joint_indices = get_indices_from_substring(
            robot.left_yaw_roll_joints, sorted_joint_names
        ).to(self.device)
        self.right_yaw_roll_joint_indices = get_indices_from_substring(
            robot.right_yaw_roll_joints, sorted_joint_names
        ).to(self.device)
        self.upper_body_joint_indices = get_indices_from_substring(robot.upper_body_joints, sorted_joint_names).to(
            self.device
        )
        self.waist_joint_indices = get_indices_from_substring(robot.waist_joints, sorted_joint_names).to(self.device)
        return super()._init_joint_cfg()

    def get_phase(self):
        """Return normalized gait phase in [0, 1)."""
        feet_cycle_time = self.cfg.rewards.extras.feet_cycle_time
        phase = (self._episode_steps * self.step_dt) % feet_cycle_time / feet_cycle_time
        return phase

    def _compute_ref_state(self, envstate):
        """Compute reference target position for walking task."""
        phase = self.get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        sin_pos_l = sin_pos.clone()
        sin_pos_r = -sin_pos.clone()
        self.ref_dof_pos = self.default_dof_pos.clone().unsqueeze(0).repeat(self.num_envs, 1)

        # Scale gait amplitude by command magnitude so zero command => no gait
        lin_speed = torch.norm(self.commands[:, :2], dim=1)
        yaw_speed = torch.abs(self.commands[:, 2]) if self.commands.shape[1] > 2 else 0.0
        speed_factor = torch.clamp(lin_speed + 0.5 * yaw_speed, 0.0, 1.0).unsqueeze(1)

        scale_1 = self.cfg.rewards.extras.target_joint_pos_scale
        scale_2 = 2 * scale_1
        # sin_pos_l[sin_pos_l > 0] = 0
        self.ref_dof_pos[:, self.left_hip_pitch_joint_idx] = sin_pos_l * scale_1
        self.ref_dof_pos[:, self.left_knee_joint_idx] = sin_pos_l * scale_2
        self.ref_dof_pos[:, self.left_ankle_joint_idx] = sin_pos_l * scale_1
        # sin_pos_r[sin_pos_r > 0] = 0
        self.ref_dof_pos[:, self.right_hip_pitch_joint_idx] = sin_pos_r * scale_1
        self.ref_dof_pos[:, self.right_knee_joint_idx] = sin_pos_r * scale_2
        self.ref_dof_pos[:, self.right_ankle_joint_idx] = sin_pos_r * scale_1

        # Double support phase
        self.ref_dof_pos[torch.abs(sin_pos) < 0.1] = 0
        self.ref_dof_pos *= speed_factor
        self.ref_dof_pos = self.ref_dof_pos.clamp(self.dof_pos_limits[:, 0], self.dof_pos_limits[:, 1])
        envstate.robots[self.robot.name].extra["phase"] = phase
        envstate.robots[self.robot.name].extra["ref_dof_pos"] = self.ref_dof_pos

    def _parse_state_for_rewards(self, env_states):
        """Parse necessary state components for reward calculation."""
        self._compute_ref_state(env_states)
        return super()._parse_state_for_rewards(env_states)

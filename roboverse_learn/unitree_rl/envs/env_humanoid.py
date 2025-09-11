import torch

from typing import Union
from roboverse_pack.robots import G1Dof12Cfg
from roboverse_learn.unitree_rl.helper.utils import get_indices_from_substring

from .env_legged_robot import LeggedRobotEnv

class HumanoidEnv(LeggedRobotEnv):
    def _init_rigid_body_indices(self):
        robot: Union[G1Dof12Cfg] = self.robot
        sorted_body_names: list[str] = self.sorted_body_names
        self.knee_indices = get_indices_from_substring(robot.knee_links, sorted_body_names).to(self.device)
        self.elbow_indices = get_indices_from_substring(robot.elbow_links, sorted_body_names).to(self.device)
        self.wrist_indices = get_indices_from_substring(robot.wrist_links, sorted_body_names).to(self.device)
        self.torso_indices = get_indices_from_substring(robot.torso_links, sorted_body_names).to(self.device)
        return super()._init_rigid_body_indices()

    def _init_joint_cfg(self):
        robot: Union[G1Dof12Cfg] = self.robot
        sorted_joint_names: list[str] = self.sorted_joint_names
        self.left_yaw_roll_joint_indices = get_indices_from_substring(robot.left_yaw_roll_joints, sorted_joint_names).to(self.device)
        self.right_yaw_roll_joint_indices = get_indices_from_substring(robot.right_yaw_roll_joints, sorted_joint_names).to(self.device)
        self.upper_body_joint_indices = get_indices_from_substring(robot.upper_body_joints, sorted_joint_names).to(self.device)
        self.waist_joint_indices = get_indices_from_substring(robot.waist_joints, sorted_joint_names).to(self.device)
        return super()._init_joint_cfg()

    def _init_buffers(self):
        self.leg_phase = torch.zeros(size=(self.num_envs, len(self.feet_indices)), dtype=torch.bool, device=self.device)
        return super()._init_buffers()

    def _post_physics_step_callback(self):
        self._update_feet_gait()
        return super()._post_physics_step_callback()

    def _update_feet_gait(self):
        """Add phase into states"""
        phase = self._get_feet_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        # left foot stance
        self.leg_phase[:, 0] = sin_pos >= 0
        # right foot stance
        self.leg_phase[:, 1] = sin_pos < 0
        # Double support phase
        self.leg_phase[torch.abs(sin_pos) < self.cfg.rewards.extras.all_feet_contact_time / 2.0] = True

    def _get_feet_phase(self):
        feet_cycle_time = self.cfg.rewards.extras.feet_cycle_time
        phase = (self.episode_steps * self.dt) % feet_cycle_time / feet_cycle_time
        return phase

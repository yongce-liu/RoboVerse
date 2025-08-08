from __future__ import annotations

import torch

from roboverse_learn.unitree_rl.configs.base_humanoid import BaseHumanoidCfg
from roboverse_learn.unitree_rl.envs.base_legged import LeggedRobot
from roboverse_learn.unitree_rl.utils import get_body_reindexed_indices_from_substring, get_joint_reindexed_indices_from_substring

from metasim.utils.state import TensorState
from metasim.cfg.scenario import ScenarioCfg
from metasim.utils.humanoid_robot_util import contact_forces_tensor


class Humanoid(LeggedRobot):
    """
    Inherit from LeggedRobot to implement a humanoid robot environment.
    The main difference is the additional joints and rigid bodies specific to humanoid robots, e.g., knees, elbows, wrists, and torso.
    """
    cfg: BaseHumanoidCfg

    def __init__(self, scenario: ScenarioCfg):
        super().__init__(scenario)
        self._parse_joint_indices(scenario.robots[0]) # new funcs for utilies

    # region: Parse configs & Get the necessary parametres
    def _parse_rigid_body_indices(self, robot):
        """
        Parse rigid body indices from robot cfg.
        """
        # parse for foot. termination_contact, penalised_contact
        super()._parse_rigid_body_indices(robot)
        knee_names = robot.knee_links
        elbow_names = robot.elbow_links
        wrist_names = robot.wrist_links
        torso_names = robot.torso_links

        # get sorted indices for specific body links
        self.knee_indices = get_body_reindexed_indices_from_substring(
            self.env.handler, robot.name, knee_names, device=self.device
        )
        self.elbow_indices = get_body_reindexed_indices_from_substring(
            self.env.handler, robot.name, elbow_names, device=self.device
        )
        self.wrist_indices = get_body_reindexed_indices_from_substring(
            self.env.handler, robot.name, wrist_names, device=self.device
        )
        self.torso_indices = get_body_reindexed_indices_from_substring(
            self.env.handler, robot.name, torso_names, device=self.device
        )

        # attach to cfg for reward computation.
        self.cfg.knee_indices = self.knee_indices
        self.cfg.elbow_indices = self.elbow_indices
        self.cfg.wrist_indices = self.wrist_indices
        self.cfg.torso_indices = self.torso_indices

    def _parse_joint_indices(self, robot):
        """
        Parse joint indices.
        """
        left_yaw_roll_names = robot.left_yaw_roll_joints
        right_yaw_roll_names = robot.right_yaw_roll_joints
        upper_body_names = robot.upper_body_joints
        self.cfg.left_yaw_roll_joint_indices = get_joint_reindexed_indices_from_substring(
            self.env.handler, robot.name, left_yaw_roll_names, device=self.device
        )
        self.cfg.right_yaw_roll_joint_indices = get_joint_reindexed_indices_from_substring(
            self.env.handler, robot.name, right_yaw_roll_names, device=self.device
        )
        self.cfg.upper_body_joint_indices = get_joint_reindexed_indices_from_substring(
            self.env.handler, robot.name, upper_body_names, device=self.device
        )
    # endregion

    # region: Parse states for reward computation
    def _parse_foot_all(self, envstate: TensorState):
        """
        Run all the parse foot function sequentially. foot pos update must run first.

        Note that orders matters here, since some of the foot states are computed based on the previous foot states.
        """
        super()._parse_foot_all(envstate)
        self._parse_feet_clearance(envstate)

    def _parse_feet_clearance(self, envstate: TensorState):
        """Calculates reward based on the clearance of the swing leg from the ground during movement.
        Encourages appropriate lift of the feet during the swing phase of the gait.


        Directly calculates reward since no intermediate variables are reused for other reward.
        """
        contact = contact_forces_tensor(envstate, self.robot.name)[:, self.feet_indices, 2] > 5.0
        feet_z = envstate.robots[self.robot.name].body_state[:, self.feet_indices, 2] - 0.05
        delta_z = feet_z - self.last_feet_z
        self.feet_height += delta_z
        self.last_feet_z = feet_z

        # Compute swing mask
        swing_mask = 1 - self._get_gait_phase()

        # feet height should be closed to target feet height at the peak
        rew_pos = torch.abs(self.feet_height - self.cfg.reward_cfg.target_feet_height) < 0.01
        rew_pos = torch.sum(rew_pos * swing_mask, dim=1)
        self.feet_height *= ~contact
        envstate.robots[self.robot.name].extra["feet_clearance"] = rew_pos
    # endregion

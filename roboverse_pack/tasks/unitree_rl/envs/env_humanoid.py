from __future__ import annotations

from roboverse_learn.rl.unitree_rl.helper import get_indices_from_substring

from .env_legged_robot import LeggedRobotEnv


class HumanoidEnv(LeggedRobotEnv):
    """Humanoid specializations on top of LeggedRobotEnv."""

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
        phase = (self.episode_steps * self.dt) % feet_cycle_time / feet_cycle_time
        return phase

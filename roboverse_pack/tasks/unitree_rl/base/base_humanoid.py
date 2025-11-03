from __future__ import annotations

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

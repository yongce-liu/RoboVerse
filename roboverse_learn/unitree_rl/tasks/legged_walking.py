from __future__ import annotations

from typing import Callable
import torch

from metasim.cfg.scenario import ScenarioCfg
from metasim.utils.state import TensorState
from metasim.utils import configclass
from metasim.utils.humanoid_robot_util import (
    contact_forces_tensor,
    dof_pos_tensor,
    dof_vel_tensor,
    ref_dof_pos_tensor,
)

from roboverse_learn.unitree_rl.envs.base_legged import LeggedRobot
from roboverse_learn.unitree_rl.configs.base_legged import BaseLeggedTaskCfg, LeggedRobotCfgPPO
from roboverse_learn.unitree_rl.configs import reward_funcs as rfs


# Training Config
@configclass
class LeggedWalkingCfgPPO(LeggedRobotCfgPPO):
    algorithm = LeggedRobotCfgPPO.Algorithm(entropy_coef = 0.01)
    runner = LeggedRobotCfgPPO.Runner(experiment_name = "legged_walking")


# Config
@configclass
class LeggedWalkingCfg(BaseLeggedTaskCfg):
    """Configuration for the walking task."""
    task_name = "legged_walking"

    ppo_cfg = LeggedWalkingCfgPPO()

    frame_stack = 1
    c_frame_stack = 3

    reward_functions: list[Callable] = [
        rfs.reward_lin_vel_z,
        rfs.reward_ang_vel_xy,
        rfs.reward_orientation,
        rfs.reward_base_height,
        rfs.reward_torques,
        rfs.reward_dof_vel,
        rfs.reward_dof_acc,
        rfs.reward_action_rate,
        rfs.reward_collision,
        rfs.reward_termination,
        rfs.reward_dof_pos_limits,
        rfs.reward_tracking_lin_vel,
        rfs.reward_tracking_ang_vel,
        rfs.reward_feet_air_time,
        rfs.reward_stumble,
        rfs.reward_stand_still,
        # reward_dof_vel_limits,
        # reward_torque_limits,
        # reward_feet_contact_forces,
    ]

    reward_weights: dict[str, float] = {
        "termination": -0.0,
        "tracking_lin_vel": 1.0,
        "tracking_ang_vel": 0.5,
        "lin_vel_z": -2.0,
        "ang_vel_xy": -0.05,
        "orientation": -0.0,
        "torques": -0.00001,
        "dof_vel": -0.0,
        "dof_acc": -2.5e-7,
        "base_height": -0.0,
        "feet_air_time":  1.0,
        "collision": -1.0,
        "feet_stumble": -0.0,
        "action_rate": -0.01,
        "stand_still": -0.0,
        "torques": -0.0002,
        "dof_pos_limits": -10.0
    }

    def __post_init__(self):
        super().__post_init__()
        self.num_single_obs: int = self.commands.commands_dim + 9  + 3 * self.num_actions #
        self.num_observations: int = int(self.frame_stack * self.num_single_obs)
        self.single_num_privileged_obs: int = 0
        self.num_privileged_obs = int(self.c_frame_stack * self.single_num_privileged_obs)


# Environment
class LeggedWalkingTask(LeggedRobot):
    """
    Wrapper for walking task for legged robots.
    """
    def __init__(self, scenario: ScenarioCfg):
        super().__init__(scenario)

    def _init_buffers(self):
        super()._init_buffers()
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)

    def _get_noise_scale_vec(self, cfg: BaseLeggedTaskCfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = cfg.noise.add_noise
        noise_scales = cfg.noise.noise_scales
        noise_level = cfg.noise.noise_level
        noise_vec[0:3] = 0. # commands
        noise_vec[3:6] = noise_scales.lin_vel * noise_level * cfg.normalization.obs_scales.lin_vel
        noise_vec[6:9] = noise_scales.ang_vel * noise_level * cfg.normalization.obs_scales.ang_vel
        noise_vec[9:12] = noise_scales.gravity * noise_level
        noise_vec[12:12+self.num_actions] = noise_scales.dof_pos * noise_level * cfg.normalization.obs_scales.dof_pos
        noise_vec[12+self.num_actions:12+2*self.num_actions] = noise_scales.dof_vel * noise_level * cfg.normalization.obs_scales.dof_vel
        noise_vec[12+2*self.num_actions:] = 0. # previous actions

        return noise_vec

    def compute_observations(self, envstate: TensorState):
        """compute observations and priviledged observation"""
        q = (dof_pos_tensor(envstate, self.robot.name) - self.cfg.default_joint_pd_target
        ) * self.cfg.normalization.obs_scales.dof_pos
        dq = dof_vel_tensor(envstate, self.robot.name) * self.cfg.normalization.obs_scales.dof_vel

        self.obs_buf = torch.cat((
                        self.commands[:, :3] * self.commands_scale,
                        self.base_lin_vel * self.cfg.normalization.obs_scales.lin_vel, # 3
                        self.base_ang_vel  * self.cfg.normalization.obs_scales.ang_vel, # 3
                        self.projected_gravity, # 3
                        q,
                        dq,
                        self.actions,
                        ),dim=-1)

        # add perceptive inputs if not blind
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

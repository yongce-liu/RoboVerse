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
from roboverse_learn.unitree_rl.configs.reward_funcs import (
    reward_action_rate,
    reward_ang_vel_xy,
    reward_base_height,
    reward_collision,
    reward_dof_acc,
    reward_dof_pos_limits,
    reward_dof_vel,
    reward_dof_vel_limits,
    reward_feet_air_time,
    reward_feet_contact_forces,
    reward_lin_vel_z,
    reward_orientation,
    reward_stand_still,
    reward_stumble,
    reward_termination,
    reward_torque_limits,
    reward_torques,
    reward_tracking_ang_vel,
    reward_tracking_lin_vel,
)


@configclass
class LeggedWalkingCfgPPO(LeggedRobotCfgPPO):
    seed: int = 0

    @configclass
    class Runner(LeggedRobotCfgPPO.Runner):
        num_steps_per_env = 60
        max_iterations = 15001
        save_interval = 500
        experiment_name = "legged_walking"

    runner: Runner = Runner()


@configclass
class LeggedWalkingCfg(BaseLeggedTaskCfg):
    """Configuration for the walking task."""
    task_name = "legged_walking"

    ppo_cfg = LeggedWalkingCfgPPO()

    command_dim = 3
    frame_stack = 1
    c_frame_stack = 3
    commands = BaseLeggedTaskCfg.CommandsConfig(num_commands=4, resampling_time=8.0)

    reward_functions: list[Callable] = [
        reward_lin_vel_z,
        reward_ang_vel_xy,
        reward_orientation,
        reward_base_height,
        reward_torques,
        reward_dof_vel,
        reward_dof_acc,
        reward_action_rate,
        reward_collision,
        reward_termination,
        reward_dof_pos_limits,
        reward_dof_vel_limits,
        reward_torque_limits,
        reward_tracking_lin_vel,
        reward_tracking_ang_vel,
        reward_feet_air_time,
        reward_stumble,
        reward_stand_still,
        reward_feet_contact_forces
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
        "feet_air_time": 1.0,
        "collision": -1.0,
        "feet_stumble": -0.0,
        "action_rate": -0.01,
        "stand_still": -0.0,
    }

    def __post_init__(self):
        super().__post_init__()
        self.num_single_obs: int = 9 + self.command_dim + 3 * self.num_actions   #
        self.num_observations: int = int(self.frame_stack * self.num_single_obs)
        self.single_num_privileged_obs: int = 4 * self.num_actions + 25
        self.num_privileged_obs = int(self.c_frame_stack * self.single_num_privileged_obs)


class LeggedWalkingTask(LeggedRobot):
    """
    Wrapper for walking task for legged robots.
    """
    def __init__(self, scenario: ScenarioCfg):
        super().__init__(scenario)

    def _init_buffers(self):
        super()._init_buffers()
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)

    def compute_observations(self, envstate: TensorState):
        """compute observations and priviledged observation"""
        q = (
            dof_pos_tensor(envstate, self.robot.name) - self.cfg.default_joint_pd_target
        ) * self.cfg.normalization.obs_scales.dof_pos
        dq = dof_vel_tensor(envstate, self.robot.name) * self.cfg.normalization.obs_scales.dof_vel

        self.obs_buf = torch.cat((
                        self.commands[:, :3] * self.commands_scale,
                        self.base_lin_vel * self.cfg.normalization.obs_scales.lin_vel,
                        self.base_ang_vel  * self.cfg.normalization.obs_scales.ang_vel,
                        self.projected_gravity,
                        q,
                        dq,
                        self.actions
                        ),dim=-1)
        # add perceptive inputs if not blind
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def _get_noise_scale_vec(self, cfg):
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

from __future__ import annotations

from typing import Callable

import torch

from metasim.cfg.control import ControlCfg
from metasim.cfg.scenario import ScenarioCfg
from metasim.utils import configclass
from metasim.utils.humanoid_robot_util import (
    dof_pos_tensor,
    dof_vel_tensor,
)
from roboverse_learn.unitree_rl.configs import reward_funcs as rfs
from roboverse_learn.unitree_rl.configs.base_legged import BaseLeggedTaskCfg, LeggedRobotCfgPPO
from roboverse_learn.unitree_rl.envs.base_humanoid import Humanoid


# Training Config
@configclass
class Dof12WalkingCfgPPO(LeggedRobotCfgPPO):
    policy = LeggedRobotCfgPPO.Policy(
        init_noise_std=0.8,
        actor_hidden_dims=[32],
        critic_hidden_dims=[32],
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_size=64,
        rnn_num_layers=1,
    )
    algorithm = LeggedRobotCfgPPO.Algorithm(entropy_coef=0.01)
    runner = LeggedRobotCfgPPO.Runner(
        experiment_name="dof12_walking",
        policy_class_name="ActorCriticRecurrent",
        max_iterations=15001,
        save_interval=50,
    )


# Config
@configclass
class Dof12WalkingCfg(BaseLeggedTaskCfg):
    """Configuration for the walking task."""

    task_name = "dof12_walking"

    ppo_cfg = Dof12WalkingCfgPPO()

    control = ControlCfg(action_scale=0.25, action_offset=True, torque_limit_scale=0.85)

    frame_stack = 1
    c_frame_stack = 1

    reward_cfg = BaseLeggedTaskCfg.RewardCfg(
        base_height_target=0.75, soft_dof_pos_limit=0.9, cycle_time=0.8, target_feet_height=0.08
    )

    reward_functions: list[Callable] = [
        rfs.reward_tracking_lin_vel,
        rfs.reward_tracking_ang_vel,
        rfs.reward_lin_vel_z,
        rfs.reward_ang_vel_xy,
        rfs.reward_orientation,
        rfs.reward_orientation_sq,
        rfs.reward_base_height,
        rfs.reward_base_height_sq,
        rfs.reward_dof_acc,
        rfs.reward_dof_vel,
        rfs.reward_feet_air_time,
        rfs.reward_collision,
        rfs.reward_action_rate,
        rfs.reward_dof_pos_limits,
        rfs.reward_hip_pos,
        rfs.reward_contact_no_vel,
        rfs.reward_feet_clearance,
        rfs.reward_feet_contact_number,
        rfs.reward_alive,
        rfs.reward_feet_swing_height,
        rfs.reward_contact,
        rfs.reward_torques,
    ]

    reward_weights: dict[str, float] = {
        "termination": -0.0,
        "feet_stumble": -0.0,
        "tracking_lin_vel": 1.0,
        "tracking_ang_vel": 0.5,
        "lin_vel_z": -2.0,
        "ang_vel_xy": -0.05,
        # "orientation": -1.0,
        "orientation_sq": -1.0,
        # "base_height": 10.0,
        "base_height_sq": -10.0,
        "dof_acc": -2.5e-7,
        "dof_vel": -1e-3,
        "feet_air_time": 0.0,
        "collision": 0.0,
        "action_rate": -0.01,
        "dof_pos_limits": -5.0,
        "alive": 0.15,
        "hip_pos": -1.0,
        "contact_no_vel": -0.2,
        "feet_swing_height": -20.0,
        # "feet_clearance": 2.0,
        "contact": 0.18,
        # "feet_contact_number": 2.4,
        "torques": -0.00001,
    }

    def __post_init__(self):
        super().__post_init__()
        self.num_single_obs: int = 6 + self.commands.commands_dim + 3 * self.num_actions + 2
        self.num_observations: int = int(self.frame_stack * self.num_single_obs)
        self.single_num_privileged_obs: int = 9 + self.commands.commands_dim + 3 * self.num_actions + 2
        self.num_privileged_obs = int(self.c_frame_stack * self.single_num_privileged_obs)


# Environment
class Dof12WalkingTask(Humanoid):
    """
    Wrapper for walking task for legged robots.
    """

    def __init__(self, scenario: ScenarioCfg):
        super().__init__(scenario)

    def _init_buffers(self):
        super()._init_buffers()
        self.noise_scale_vec = self._get_noise_scale_vec()

    def _get_noise_scale_vec(self):
        """Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.cfg.normalization.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = 0.0  # commands
        noise_vec[9 : 9 + self.num_actions] = (
            noise_scales.dof_pos * noise_level * self.cfg.normalization.obs_scales.dof_pos
        )
        noise_vec[9 + self.num_actions : 9 + 2 * self.num_actions] = (
            noise_scales.dof_vel * noise_level * self.cfg.normalization.obs_scales.dof_vel
        )
        noise_vec[9 + 2 * self.num_actions : 9 + 3 * self.num_actions] = 0.0  # previous actions
        noise_vec[9 + 3 * self.num_actions : 9 + 3 * self.num_actions + 2] = 0.0  # sin/cos phase

        return noise_vec

    def compute_observations(self, envstate):
        """Compute observations using the simplified layout requested by the user.

        Layout (concatenated in this exact order):
            1. base_ang_vel (scaled)        - 3 dims
            2. projected_gravity            - 3 dims
            3. commands (first 3, scaled)   - 3 dims
            4. dof_pos deviation (scaled)  - |A| dims
            5. dof_vel (scaled)            - |A| dims
            6. previous actions            - |A| dims
            7. sin(phase)                  - 1 dim
            8. cos(phase)                  - 1 dim
        """
        # --- Phase features
        phase = self._get_phase()
        sin_phase = torch.sin(2 * torch.pi * phase).unsqueeze(1)
        cos_phase = torch.cos(2 * torch.pi * phase).unsqueeze(1)
        # --- Joint position / velocity (normalised) in NATIVE simulator order
        q = (
            dof_pos_tensor(envstate, self.robot.name) - self.cfg.default_joint_pd_target
        ) * self.cfg.normalization.obs_scales.dof_pos
        dq = dof_vel_tensor(envstate, self.robot.name) * self.cfg.normalization.obs_scales.dof_vel

        # --- Assemble observation buffer
        self.privileged_obs_buf = torch.cat(
            (
                self.base_lin_vel * self.cfg.normalization.obs_scales.lin_vel,
                self.base_ang_vel * self.cfg.normalization.obs_scales.ang_vel,
                self.projected_gravity,
                self.commands[:, :3] * self.commands_scale,
                q,
                dq,
                self.actions,
                sin_phase,
                cos_phase,
            ),
            dim=-1,
        )

        obs_buf = torch.cat(
            (
                self.base_ang_vel * self.cfg.normalization.obs_scales.ang_vel,  # 3
                self.projected_gravity,  # 3
                self.commands[:, :3] * self.commands_scale,  # 3
                q,  # num_actions
                dq,  # num_actions
                self.actions,  # num_actions
                sin_phase,  # 1
                cos_phase,  # 1
            ),
            dim=-1,
        )
        # print(self.cfg.default_joint_pd_target), input()
        # Frame stacking (reuse existing obs_history)
        obs_now = obs_buf.clone()
        self.obs_history.append(obs_now)
        self.critic_history.append(self.privileged_obs_buf)
        obs_buf_all = torch.stack([self.obs_history[i] for i in range(self.obs_history.maxlen)], dim=1)
        self.obs_buf = obs_buf_all.reshape(self.num_envs, -1)
        self.privileged_obs_buf = torch.cat([self.critic_history[i] for i in range(self.cfg.c_frame_stack)], dim=1)

        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

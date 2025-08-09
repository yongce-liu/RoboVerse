from __future__ import annotations

from typing import Callable
from functools import partial
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

from roboverse_learn.unitree_rl.utils import find_unique_candidate
from roboverse_learn.unitree_rl.envs.base_humanoid import Humanoid
from roboverse_learn.unitree_rl.envs.base_legged import LeggedRobot
from roboverse_learn.unitree_rl.configs.base_legged import BaseLeggedTaskCfg, LeggedRobotCfgPPO
from roboverse_learn.unitree_rl.configs.base_humanoid import BaseHumanoidCfg
from roboverse_learn.unitree_rl.configs import reward_funcs as rfs


@configclass
class HumanoidWalkingCfgPPO(LeggedRobotCfgPPO):
    seed: int = 0

    algorithm = LeggedRobotCfgPPO.Algorithm(
        entropy_coef=0.001, learning_rate=1e-5, num_learning_epochs=2, gamma=0.994, lam=0.9
    )
    runner = LeggedRobotCfgPPO.Runner(num_steps_per_env = 60,
                                      max_iterations = 15001,
                                      save_interval = 500,
                                      experiment_name = "humanoid_walking")


@configclass
class HumanoidWalkingCfg(BaseHumanoidCfg):
    """Configuration for the walking task."""
    task_name = "humanoid_walking"

    ppo_cfg = HumanoidWalkingCfgPPO()

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
        rfs.reward_dof_vel_limits,
        rfs.reward_torque_limits,
        rfs.reward_tracking_lin_vel,
        rfs.reward_tracking_ang_vel,
        rfs.reward_feet_air_time,
        rfs.reward_stumble,
        rfs.reward_stand_still,
        rfs.reward_feet_contact_forces,
        rfs.reward_joint_pos,
        rfs.reward_feet_distance,
        rfs.reward_knee_distance,
        rfs.reward_elbow_distance,
        rfs.reward_foot_slip,
        rfs.reward_feet_contact_number,
        rfs.reward_default_joint_pos,
        rfs.reward_upper_body_pos,
        rfs.reward_base_acc,
        rfs.reward_vel_mismatch_exp,
        rfs.reward_track_vel_hard,
        rfs.reward_feet_clearance,
        rfs.reward_low_speed,
        rfs.reward_action_smoothness,
    ]

    reward_weights: dict[str, float] = {
        "termination": -0.0,
        "lin_vel_z": -0.0,
        "ang_vel_xy": -0.05,
        "base_height": 0.2,
        "feet_air_time": 1.0,
        "collision": -1.0,
        "feet_stumble": -0.0,
        "stand_still": -0.0,
        "joint_pos": 1.6,
        "feet_clearance": 2.0,
        "feet_contact_number": 2.4,
        # gait
        "foot_slip": -0.05,
        "feet_distance": 0.2,
        "knee_distance": 0.2,
        # contact
        "feet_contact_forces": -0.01,
        # vel tracking
        "tracking_lin_vel": 2.4,
        "tracking_ang_vel": 2.2,
        "vel_mismatch_exp": 0.5,
        "low_speed": 0.2,
        "track_vel_hard": 1.0,
        # base pos
        "default_joint_pos": 1.0,
        "upper_body_pos": 0.5,
        "orientation": 1.0,
        "base_acc": 0.2,
        # energy
        "action_smoothness": -0.002,
        "torques": -1e-5,
        "dof_vel": -5e-4,
        "dof_acc": -1e-7,
        "torque_limits": 0.001,
        "dof_pos_limits": -0.01,
        # optional
        "action_rate": -0.0,
    }

    def __post_init__(self):
        super().__post_init__()
        self.num_single_obs: int = self.commands.commands_dim + 9 + 3 * self.num_actions + 2
        self.num_observations: int = int(self.frame_stack * self.num_single_obs)
        self.single_num_privileged_obs: int = self.commands.commands_dim + 12 + 4 * self.num_actions + 14
        self.num_privileged_obs = int(self.c_frame_stack * self.single_num_privileged_obs)


class HumanoidWalkingTask(Humanoid):
    """
    Wrapper for walking

    # TODO implement push robot
    """

    def __init__(self, scenario: ScenarioCfg):
        super().__init__(scenario)
        self._prepare_ref_indices()

    def _init_buffers(self):
        super()._init_buffers()
        self.noise_scale_vec = self._get_noise_scale_vec()

    def _prepare_ref_indices(self):
        """get joint indices for reference pos computation."""
        joint_names = self.env.handler.get_joint_names(self.robot.name)
        find_func = partial(find_unique_candidate, data_base=joint_names)
        name_extend_func = lambda x: [x, f"{x}_joint"]
        self.left_hip_pitch_joint_idx = find_func(candidates=name_extend_func("left_hip_pitch"))
        self.left_knee_joint_idx = find_func(candidates=name_extend_func("left_knee"))
        self.right_hip_pitch_joint_idx = find_func(candidates=name_extend_func("right_hip_pitch"))
        self.right_knee_joint_idx = find_func(candidates=name_extend_func("right_knee"))
        self.left_ankle_joint_idx = find_func(candidates=name_extend_func("left_ankle")+name_extend_func("left_ankle_pitch"))
        self.right_ankle_joint_idx = find_func(candidates=name_extend_func("right_ankle")+name_extend_func("right_ankle_pitch"))

    def _compute_ref_state(self):
        """compute reference target position for walking task."""
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        sin_pos_l = sin_pos.clone()
        sin_pos_r = sin_pos.clone()
        self.ref_dof_pos = torch.zeros(
            self.num_envs, self.env.handler.robot_num_dof, device=self.device, requires_grad=False
        )
        scale_1 = self.cfg.reward_cfg.target_joint_pos_scale
        scale_2 = 2 * scale_1
        sin_pos_l[sin_pos_l > 0] = 0
        self.ref_dof_pos[:, self.left_hip_pitch_joint_idx] = sin_pos_l * scale_1  # left_hip_pitch_joint
        self.ref_dof_pos[:, self.left_knee_joint_idx] = sin_pos_l * scale_2  # left_knee_joint
        self.ref_dof_pos[:, self.left_ankle_joint_idx] = sin_pos_l * scale_1  # left_ankle_joint
        sin_pos_r[sin_pos_r < 0] = 0
        self.ref_dof_pos[:, self.right_hip_pitch_joint_idx] = sin_pos_r * scale_1  # right_hip_pitch_joint
        self.ref_dof_pos[:, self.right_knee_joint_idx] = sin_pos_r * scale_2  # right_knee_joint
        self.ref_dof_pos[:, self.right_ankle_joint_idx] = sin_pos_r * scale_1  # right_ankle_joint
        # Double support phase
        self.ref_dof_pos[torch.abs(sin_pos) < 0.1] = 0
        self.ref_dof_pos = 2 * self.ref_dof_pos

    def _parse_ref_pos(self, envstate):
        envstate.robots[self.robot.name].extra["ref_dof_pos"] = self.ref_dof_pos

    def _parse_state_for_reward(self, envstate):
        """
        Parse all the states to prepare for reward computation, legged_robot level reward computation.
        """

        super()._parse_state_for_reward(envstate)
        self._compute_ref_state()
        self._parse_ref_pos(envstate)

    def _get_noise_scale_vec(self) -> torch.Tensor:
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[0:3] = 0. # commands
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.cfg.normalization.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:9+self.num_actions] = noise_scales.dof_pos * noise_level * self.cfg.normalization.obs_scales.dof_pos
        noise_vec[9+self.num_actions:9+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.cfg.normalization.obs_scales.dof_vel
        noise_vec[9+2*self.num_actions:9+3*self.num_actions] = 0. # previous actions
        noise_vec[9+3*self.num_actions:9+3*self.num_actions+2] = 0. # sin/cos phase

        return noise_vec

    def compute_observations(self, envstates):
        """compute observation and privileged observation."""

        phase = self._get_phase()

        sin_phase = torch.sin(2 * torch.pi * phase).unsqueeze(1)
        cos_phase = torch.cos(2 * torch.pi * phase).unsqueeze(1)

        stance_mask = self._get_gait_phase()
        contact_mask = contact_forces_tensor(envstates, self.robot.name)[:, self.feet_indices, 2] > 5

        q = (
            dof_pos_tensor(envstates, self.robot.name) - self.cfg.default_joint_pd_target
        ) * self.cfg.normalization.obs_scales.dof_pos
        dq = dof_vel_tensor(envstates, self.robot.name) * self.cfg.normalization.obs_scales.dof_vel
        diff = dof_pos_tensor(envstates, self.robot.name) - ref_dof_pos_tensor(envstates, self.robot.name)

        self.privileged_obs_buf = torch.cat(
            (
                self.commands[:, :3] * self.commands_scale,  # 3
                self.base_lin_vel * self.cfg.normalization.obs_scales.lin_vel,  # 3
                self.base_ang_vel * self.cfg.normalization.obs_scales.ang_vel,  # 3
                self.base_euler_xyz * self.cfg.normalization.obs_scales.quat,  # 3
                self.projected_gravity,  # 3
                q,  # |A|
                dq,  # |A|
                self.actions,  # |A|
                diff,  # |A|
                self.rand_push_force[:, :3],  # 3
                self.rand_push_torque,  # 3
                self.env_frictions,  # 1
                self.body_mass / 30.0,  # 1
                stance_mask,  # 2
                contact_mask,  # 2
                sin_phase, # 1
                cos_phase, # 1
            ),
            dim=-1,
        )

        obs_buf = torch.cat(
            (
                self.commands[:, :3] * self.commands_scale,  # 3
                self.base_ang_vel * self.cfg.normalization.obs_scales.ang_vel,  # 3
                self.base_euler_xyz * self.cfg.normalization.obs_scales.quat,  # 3
                self.projected_gravity, # 3
                q,  # |A|
                dq,  # |A|
                self.actions,
                sin_phase,
                cos_phase,
            ),
            dim=-1,
        )

        obs_now = obs_buf.clone()
        self.obs_history.append(obs_now)
        self.critic_history.append(self.privileged_obs_buf)
        obs_buf_all = torch.stack([self.obs_history[i] for i in range(self.obs_history.maxlen)], dim=1)
        self.obs_buf = obs_buf_all.reshape(self.num_envs, -1)
        self.privileged_obs_buf = torch.cat([self.critic_history[i] for i in range(self.cfg.c_frame_stack)], dim=1)

        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

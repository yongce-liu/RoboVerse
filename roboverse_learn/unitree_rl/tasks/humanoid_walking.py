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
from roboverse_learn.unitree_rl.configs.base_humanoid import BaseHumanoidCfg, BaseHumanoidCfgPPO
from roboverse_learn.unitree_rl.configs.reward_funcs import (
    reward_action_rate,
    reward_action_smoothness,
    reward_ang_vel_xy,
    reward_base_acc,
    reward_base_height,
    reward_collision,
    reward_default_joint_pos,
    reward_dof_acc,
    reward_dof_pos_limits,
    reward_dof_vel,
    reward_dof_vel_limits,
    reward_elbow_distance,
    reward_feet_air_time,
    reward_feet_clearance,
    reward_feet_contact_forces,
    reward_feet_contact_number,
    reward_feet_distance,
    reward_foot_slip,
    reward_joint_pos,
    reward_knee_distance,
    reward_lin_vel_z,
    reward_low_speed,
    reward_orientation,
    reward_stand_still,
    reward_stumble,
    reward_termination,
    reward_torque_limits,
    reward_torques,
    reward_track_vel_hard,
    reward_tracking_ang_vel,
    reward_tracking_lin_vel,
    reward_upper_body_pos,
    reward_vel_mismatch_exp,
)



@configclass
class HumanoidWalkingCfgPPO(BaseHumanoidCfgPPO):
    seed: int = 0

    @configclass
    class Runner(BaseHumanoidCfgPPO.Runner):
        num_steps_per_env = 60
        max_iterations = 15001
        save_interval = 500
        experiment_name = "humanoid_walking"

    runner: Runner = Runner()

@configclass
class HumanoidWalkingCfg(BaseHumanoidCfg):
    """Configuration for the walking task."""
    task_name = "humanoid_walking"

    ppo_cfg = HumanoidWalkingCfgPPO()

    command_dim = 3
    frame_stack = 1
    c_frame_stack = 3
    commands = BaseHumanoidCfg.CommandsConfig(num_commands=4, resampling_time=8.0)

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
        reward_feet_contact_forces,
        reward_joint_pos,
        reward_feet_distance,
        reward_knee_distance,
        reward_elbow_distance,
        reward_foot_slip,
        reward_feet_contact_number,
        reward_default_joint_pos,
        reward_upper_body_pos,
        reward_base_acc,
        reward_vel_mismatch_exp,
        reward_track_vel_hard,
        reward_feet_clearance,
        reward_low_speed,
        reward_action_smoothness,
    ]

    reward_weights: dict[str, float] = {
        "termination": -0.0,
        "lin_vel_z": -0.0,
        # "ang_vel_xy": -0.05,
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
        # optional
        "action_rate": -0.0,
    }

    def __post_init__(self):
        super().__post_init__()
        self.num_single_obs: int = 3 * self.num_actions + 6 + self.command_dim  #
        self.num_observations: int = int(self.frame_stack * self.num_single_obs)
        self.single_num_privileged_obs: int = 4 * self.num_actions + 25
        self.num_privileged_obs = int(self.c_frame_stack * self.single_num_privileged_obs)

class HumanoidWalkingTask(Humanoid):
    """
    Wrapper for walking

    # TODO implement push robot
    """

    def __init__(self, scenario: ScenarioCfg):
        super().__init__(scenario)
        self._prepare_ref_indices()

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

    def compute_observations(self, envstates):
        """compute observation and privileged observation."""

        phase = self._get_phase()

        sin_pos = torch.sin(2 * torch.pi * phase).unsqueeze(1)
        cos_pos = torch.cos(2 * torch.pi * phase).unsqueeze(1)

        stance_mask = self._get_gait_phase()
        contact_mask = contact_forces_tensor(envstates, self.robot.name)[:, self.feet_indices, 2] > 5

        self.command_input = torch.cat((sin_pos, cos_pos, self.commands[:, :3] * self.commands_scale), dim=1)
        self.command_input_wo_clock = self.commands[:, :3] * self.commands_scale

        q = (
            dof_pos_tensor(envstates, self.robot.name) - self.cfg.default_joint_pd_target
        ) * self.cfg.normalization.obs_scales.dof_pos
        dq = dof_vel_tensor(envstates, self.robot.name) * self.cfg.normalization.obs_scales.dof_vel
        diff = dof_pos_tensor(envstates, self.robot.name) - ref_dof_pos_tensor(envstates, self.robot.name)

        self.privileged_obs_buf = torch.cat(
            (
                self.command_input,  # 2 + 3
                q,  # |A|
                dq,  # |A|
                self.actions,  # |A|
                diff,  # |A|
                self.base_lin_vel * self.cfg.normalization.obs_scales.lin_vel,  # 3
                self.base_ang_vel * self.cfg.normalization.obs_scales.ang_vel,  # 3
                self.base_euler_xyz * self.cfg.normalization.obs_scales.quat,  # 3
                self.rand_push_force[:, :2],  # 3
                self.rand_push_torque,  # 3
                self.env_frictions,  # 1
                self.body_mass / 30.0,  # 1
                stance_mask,  # 2
                contact_mask,  # 2
            ),
            dim=-1,
        )

        obs_buf = torch.cat(
            (
                self.command_input_wo_clock,  # 3
                q,  # |A|
                dq,  # |A|
                self.actions,
                self.base_ang_vel * self.cfg.normalization.obs_scales.ang_vel,  # 3
                self.base_euler_xyz * self.cfg.normalization.obs_scales.quat,  # 3
            ),
            dim=-1,
        )

        obs_now = obs_buf.clone()
        self.obs_history.append(obs_now)
        self.critic_history.append(self.privileged_obs_buf)
        obs_buf_all = torch.stack([self.obs_history[i] for i in range(self.obs_history.maxlen)], dim=1)
        self.obs_buf = obs_buf_all.reshape(self.num_envs, -1)
        self.privileged_obs_buf = torch.cat([self.critic_history[i] for i in range(self.cfg.c_frame_stack)], dim=1)

        self.privileged_obs_buf = torch.clip(
            self.privileged_obs_buf, -self.cfg.normalization.clip_observations, self.cfg.normalization.clip_observations
        )

from __future__ import annotations

import math
from functools import partial

import torch

from metasim.scenario.scenario import ScenarioCfg
from metasim.utils import configclass
from metasim.utils.humanoid_robot_util import (
    contact_forces_tensor,
    dof_pos_tensor,
    dof_vel_tensor,
    ref_dof_pos_tensor,
)
from roboverse_learn.rl.unitree_rl.configs.base_legged import BaseLeggedTaskCfg, LeggedRobotCfgPPO
from roboverse_learn.rl.unitree_rl.envs.base_humanoid import Humanoid
from roboverse_learn.rl.unitree_rl.helper.utils import find_unique_candidate


@configclass
class G1Dof29WalkingCfgPPO(LeggedRobotCfgPPO):
    seed: int = 0

    algorithm = LeggedRobotCfgPPO.Algorithm(
        entropy_coef=0.001, learning_rate=1e-5, num_learning_epochs=2, gamma=0.994, lam=0.9
    )
    runner = LeggedRobotCfgPPO.Runner(
        num_steps_per_env=60, max_iterations=15001, save_interval=100, experiment_name="g1_dof29_walking"
    )


@configclass
class G1Dof29WalkingCfg(BaseLeggedTaskCfg):
    """Walking task configuration for Unitree G1 29DoF (no hands)."""

    task_name = "g1_dof29_walking"
    env_spacing: float = 1.0
    max_episode_length_s: int = 24
    control = BaseLeggedTaskCfg.ControlCfg(action_scale=0.25, action_offset=True, torque_limit_scale=0.85)

    # Initial state for the specific robot (no hand joints)
    init_states = [
        {
            "objects": {},
            "robots": {
                "g1_dof29": {
                    "pos": torch.tensor([0.0, 0.0, 0.76]),
                    "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    "dof_pos": {
                        # Hips & legs
                        "left_hip_pitch_joint": -0.1,
                        "left_hip_roll_joint": 0.0,
                        "left_hip_yaw_joint": 0.0,
                        "left_knee_joint": 0.3,
                        "left_ankle_pitch_joint": -0.2,
                        "left_ankle_roll_joint": 0.0,
                        "right_hip_pitch_joint": -0.1,
                        "right_hip_roll_joint": 0.0,
                        "right_hip_yaw_joint": 0.0,
                        "right_knee_joint": 0.3,
                        "right_ankle_pitch_joint": -0.2,
                        "right_ankle_roll_joint": 0.0,
                        # Waist
                        "waist_yaw_joint": 0.0,
                        "waist_roll_joint": 0.0,
                        "waist_pitch_joint": 0.0,
                        # Shoulders & arms
                        "left_shoulder_pitch_joint": 0.0,
                        "left_shoulder_roll_joint": 0.0,
                        "left_shoulder_yaw_joint": 0.0,
                        "left_elbow_joint": 0.0,
                        "left_wrist_roll_joint": 0.0,
                        "left_wrist_pitch_joint": 0.0,
                        "left_wrist_yaw_joint": 0.0,
                        "right_shoulder_pitch_joint": 0.0,
                        "right_shoulder_roll_joint": 0.0,
                        "right_shoulder_yaw_joint": 0.0,
                        "right_elbow_joint": 0.0,
                        "right_wrist_roll_joint": 0.0,
                        "right_wrist_pitch_joint": 0.0,
                        "right_wrist_yaw_joint": 0.0,
                    },
                }
            },
        }
    ]

    ppo_cfg = G1Dof29WalkingCfgPPO()

    frame_stack = 1
    c_frame_stack = 3

    reward_cfg = BaseLeggedTaskCfg.RewardCfg(base_height_target=0.76, tracking_sigma=1 / 0.25, max_contact_force=700)

    reward_weights: dict[str, float] = {
        # task tracking (mapped to your existing tracking funcs)
        "tracking_lin_vel": 1.0,      # from track_lin_vel_xy_yaw_frame_exp
        "tracking_ang_vel": 0.5,      # from track_ang_vel_z_exp
        "alive": 0.15,                # is_alive

        # base dynamics / effort
        "lin_vel_z": -2.0,            # lin_vel_z_l2
        "ang_vel_xy": -0.05,          # ang_vel_xy_l2
        "dof_vel": -0.001,            # joint_vel_l2
        "dof_acc": -2.5e-7,           # joint_acc_l2
        "action_rate": -0.05,         # action_rate_l2
        "dof_pos_limits": -5.0,       # joint_pos_limits
        "energy": -2e-5,              # energy

        # stability
        # "hip_upright_axis": 5.0,
        "waist_joint_stability": 2.0,  # waist_joint_stability

        # robot posture
        "orientation_l2": -5.0,       # flat_orientation_l2 -> orientation_l2 (your func name)
        "base_height_sq": -10.0,      # base_height_l2 -> base_height_sq (your L2 version)

        # feet / gait
        "feet_gait": 0.5,             # feet_gait
        "foot_slip": -0.2,            # feet_slide -> foot_slip (your equivalent)
        "foot_clearance_exp": 1.0,    # foot_clearance_reward -> foot_clearance_exp (your port)

        # other contacts
        "collision": -1.0,            # undesired_contacts -> collision (your penalised contacts)
    }

    def __post_init__(self):
        super().__post_init__()
        self.num_single_obs: int = self.commands.commands_dim + 9 + 3 * self.num_actions + 2
        self.num_observations: int = int(self.frame_stack * self.num_single_obs)
        self.single_num_privileged_obs: int = self.commands.commands_dim + 12 + 4 * self.num_actions + 14
        self.num_privileged_obs = int(self.c_frame_stack * self.single_num_privileged_obs)


class G1Dof29WalkingTask(Humanoid):
    """Humanoid walking wrapper for G1 29DoF robot (no hands)."""

    def __init__(self, task_cfg: G1Dof29WalkingCfg, scenario: ScenarioCfg):
        self.decimation = scenario.decimation
        self._init_from_cfg(task_cfg)
        super().__init__(scenario)
        self._prepare_ref_indices()

    def _init_from_cfg(self, task_cfg):
        self.cfg = task_cfg
        self.num_obs = self.cfg.num_observations
        self.num_actions = self.cfg.num_actions
        self.num_privileged_obs = self.cfg.num_privileged_obs
        self.max_episode_length = math.ceil(self.cfg.max_episode_length_s / self.cfg.sim_params.dt)
        from metasim.utils.dict import class_to_dict
        self.train_cfg = class_to_dict(self.cfg.ppo_cfg)

    def _init_buffers(self):
        super()._init_buffers()
        self.noise_scale_vec = self._get_noise_scale_vec()

    def _prepare_ref_indices(self):
        """Get joint indices for reference pos computation."""
        joint_names = self.handler.get_joint_names(self.robot.name)
        find_func = partial(find_unique_candidate, data_base=joint_names)

        def name_extend_func(x):
            return [x, f"{x}_joint"]

        self.left_hip_pitch_joint_idx = find_func(candidates=name_extend_func("left_hip_pitch"))
        self.left_knee_joint_idx = find_func(candidates=name_extend_func("left_knee"))
        self.right_hip_pitch_joint_idx = find_func(candidates=name_extend_func("right_hip_pitch"))
        self.right_knee_joint_idx = find_func(candidates=name_extend_func("right_knee"))
        self.left_ankle_joint_idx = find_func(
            candidates=name_extend_func("left_ankle") + name_extend_func("left_ankle_pitch")
        )
        self.right_ankle_joint_idx = find_func(
            candidates=name_extend_func("right_ankle") + name_extend_func("right_ankle_pitch")
        )

    def _compute_ref_state(self, envstate):
        """Compute reference target position for walking task."""
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        sin_pos_l = sin_pos.clone()
        sin_pos_r = sin_pos.clone()
        self.ref_dof_pos = torch.zeros(
            self.num_envs, self.robot.num_joints, device=self.device, requires_grad=False
        )

        # Scale gait amplitude by command magnitude so zero command => no gait
        lin_speed = torch.norm(self.commands[:, :2], dim=1)
        yaw_speed = torch.abs(self.commands[:, 2]) if self.commands.shape[1] > 2 else 0.0
        speed_factor = torch.clamp(lin_speed + 0.5 * yaw_speed, 0.0, 1.0).unsqueeze(1)

        scale_1 = self.cfg.reward_cfg.target_joint_pos_scale
        scale_2 = 2 * scale_1
        sin_pos_l[sin_pos_l > 0] = 0
        self.ref_dof_pos[:, self.left_hip_pitch_joint_idx] = sin_pos_l * scale_1
        self.ref_dof_pos[:, self.left_knee_joint_idx] = sin_pos_l * scale_2
        self.ref_dof_pos[:, self.left_ankle_joint_idx] = sin_pos_l * scale_1
        sin_pos_r[sin_pos_r < 0] = 0
        self.ref_dof_pos[:, self.right_hip_pitch_joint_idx] = sin_pos_r * scale_1
        self.ref_dof_pos[:, self.right_knee_joint_idx] = sin_pos_r * scale_2
        self.ref_dof_pos[:, self.right_ankle_joint_idx] = sin_pos_r * scale_1

        # Double support phase
        self.ref_dof_pos[torch.abs(sin_pos) < 0.1] = 0
        self.ref_dof_pos = 2 * self.ref_dof_pos
        self.ref_dof_pos *= speed_factor
        envstate.robots[self.robot.name].extra["phase"] = phase

    def _parse_ref_pos(self, envstate):
        envstate.robots[self.robot.name].extra["ref_dof_pos"] = self.ref_dof_pos

    def _parse_state_for_reward(self, envstate):
        """Prepare state for reward computation."""
        super()._parse_state_for_reward(envstate)
        self._compute_ref_state(envstate)
        self._parse_ref_pos(envstate)

    def _get_noise_scale_vec(self) -> torch.Tensor:
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        # Observation layout (single frame):
        # 0:3 commands, 3:6 base_ang_vel, 6:9 base_euler_xyz, 9:12 projected_gravity,
        # 12:12+A q, 12+A:12+2A dq, 12+2A:12+3A actions, +1 sin, +1 cos
        noise_vec[0:3] = 0.0  # commands (no noise)
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.cfg.normalization.obs_scales.ang_vel
        noise_vec[6:9] = 0.0  # base_euler_xyz (keep clean)
        noise_vec[9:12] = noise_scales.gravity * noise_level
        start = 12
        A = self.num_actions
        noise_vec[start : start + A] = noise_scales.dof_pos * noise_level * self.cfg.normalization.obs_scales.dof_pos
        noise_vec[start + A : start + 2 * A] = (
            noise_scales.dof_vel * noise_level * self.cfg.normalization.obs_scales.dof_vel
        )
        noise_vec[start + 2 * A : start + 3 * A] = 0.0  # previous actions (actor already outputs noisy actions)
        noise_vec[start + 3 * A : start + 3 * A + 2] = 0.0  # sin/cos phase

        return noise_vec

    def compute_observations(self, envstates):
        """Compute observation and privileged observation."""
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
                sin_phase,  # 1
                cos_phase,  # 1
            ),
            dim=-1,
        )

        obs_buf = torch.cat(
            (
                self.commands[:, :3] * self.commands_scale,  # 3
                self.base_ang_vel * self.cfg.normalization.obs_scales.ang_vel,  # 3
                self.base_euler_xyz * self.cfg.normalization.obs_scales.quat,  # 3
                self.projected_gravity,  # 3
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

        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

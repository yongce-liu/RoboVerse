from functools import partial

import torch

from metasim.types import TensorState
from metasim.utils.math import quat_rotate_inverse
from roboverse_learn.rl.unitree_rl.helper import find_unique_candidate, get_euler_xyz
from roboverse_pack.tasks.unitree_rl.envs.env_humanoid import HumanoidEnv


class WalkHumanoidEnv(HumanoidEnv):
    """Simple humanoid walking task wrapper."""

    def _init_joint_cfg(self):
        find_func = partial(find_unique_candidate, data_base=self.sorted_joint_names)

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

        return super()._init_joint_cfg()

    def _compute_ref_state(self):
        """Compute reference target position for walking task."""
        phase = self.get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        sin_pos_l = sin_pos.clone()
        sin_pos_r = sin_pos.clone()
        self.ref_dof_pos_stability = torch.zeros(
            self.num_envs, self.robot.num_joints, device=self.device, requires_grad=False
        )

        # Scale gait amplitude by command magnitude so zero command => no gait
        lin_speed = torch.norm(self.commands[:, :2], dim=1)
        yaw_speed = torch.abs(self.commands[:, 2]) if self.commands.shape[1] > 2 else 0.0
        speed_factor = torch.clamp(lin_speed + 0.5 * yaw_speed, 0.0, 1.0).unsqueeze(1)

        # scale_1 = self.cfg.reward_cfg.target_joint_pos_scale
        scale_1 = 0.17
        scale_2 = 2 * scale_1
        sin_pos_l[sin_pos_l > 0] = 0
        self.ref_dof_pos_stability[:, self.left_hip_pitch_joint_idx] = sin_pos_l * scale_1
        self.ref_dof_pos_stability[:, self.left_knee_joint_idx] = sin_pos_l * scale_2
        self.ref_dof_pos_stability[:, self.left_ankle_joint_idx] = sin_pos_l * scale_1
        sin_pos_r[sin_pos_r < 0] = 0
        self.ref_dof_pos_stability[:, self.right_hip_pitch_joint_idx] = sin_pos_r * scale_1
        self.ref_dof_pos_stability[:, self.right_knee_joint_idx] = sin_pos_r * scale_2
        self.ref_dof_pos_stability[:, self.right_ankle_joint_idx] = sin_pos_r * scale_1

        # Double support phase
        self.ref_dof_pos_stability[torch.abs(sin_pos) < self.cfg.rewards.extras.all_feet_contact_time / 2.0] = 0
        self.ref_dof_pos_stability = 2 * self.ref_dof_pos_stability
        self.ref_dof_pos_stability *= speed_factor

    def _init_buffers(self):
        self.noise_scale_vec = self._get_noise_scale_vec()
        # self._compute_ref_state()
        self.ref_dof_pos_stability = torch.zeros(
            self.num_envs, self.robot.num_joints, device=self.device, requires_grad=False
        )
        return super()._init_buffers()

    def _post_physics_step_callback(self, env_states):
        self._compute_ref_state()
        return super()._post_physics_step_callback(env_states)

    def _get_noise_scale_vec(self) -> torch.Tensor:
        noise_vec = torch.zeros(size=(101,), dtype=torch.float, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.scales
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

    def _get_gait_phase(self):
        """Add phase into states."""
        phase = self.get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        # Add double support phase
        stance_mask = torch.zeros((self.num_envs, len(self.feet_indices)), dtype=torch.bool, device=self.device)
        # left foot stance
        stance_mask[:, 0] = sin_pos >= 0
        # right foot stance
        stance_mask[:, 1] = sin_pos < 0
        # Double support phase
        stance_mask[torch.abs(sin_pos) < self.cfg.rewards.extras.all_feet_contact_time / 2.0] = True
        return stance_mask.to(torch.bool)

    def _observation(self, env_states: TensorState):
        robot_state = env_states.robots[self.name]
        base_quat = robot_state.root_state[:, 3:7]
        base_lin_vel = quat_rotate_inverse(base_quat, robot_state.root_state[:, 7:10])
        base_ang_vel = quat_rotate_inverse(base_quat, robot_state.root_state[:, 10:13])
        base_euler_xyz = get_euler_xyz(base_quat)
        projected_gravity = quat_rotate_inverse(base_quat, self.gravity_vec)

        phase = self.get_phase()
        sin_phase = torch.sin(2 * torch.pi * phase).unsqueeze(1)
        cos_phase = torch.cos(2 * torch.pi * phase).unsqueeze(1)

        stance_mask = self._get_gait_phase()
        contact_mask = env_states.extras["contact_forces"][self.name][:, self.feet_indices, 2] > 1.0

        q = (env_states.robots[self.name].joint_pos - self.default_dof_pos) * self.cfg.normalization.obs_scales.dof_pos
        dq = env_states.robots[self.name].joint_vel * self.cfg.normalization.obs_scales.dof_vel

        obs_buf = torch.cat(
            (
                self.commands[:, :3] * self.commands_scale,  # 3
                base_ang_vel * self.cfg.normalization.obs_scales.ang_vel,  # 3
                base_euler_xyz * self.cfg.normalization.obs_scales.quat,  # 3
                projected_gravity,  # 3
                q,  # |A|
                dq,  # |A|
                self.actions,
                sin_phase,
                cos_phase,
            ),
            dim=-1,
        )

        # add noise if needed
        if self.add_noise:
            obs_buf += (2 * torch.rand_like(obs_buf) - 1) * self.noise_scale_vec

        priv_obs_buf = torch.cat(
            (
                self.commands[:, :3] * self.commands_scale,  # 3
                base_lin_vel * self.cfg.normalization.obs_scales.lin_vel,  # 3
                base_ang_vel * self.cfg.normalization.obs_scales.ang_vel,  # 3
                base_euler_xyz * self.cfg.normalization.obs_scales.quat,  # 3
                projected_gravity,  # 3
                q,  # |A|
                dq,  # |A|
                self.actions,  # |A|
                env_states.robots[self.name].joint_pos - self.ref_dof_pos_stability,  # |A|
                # self.rand_push_force[:, :3],  # 3
                # self.rand_push_torque,  # 3
                # self.env_frictions,  # 1
                # self.body_mass / 30.0,  # 1
                stance_mask,  # 2
                contact_mask,  # 2
                sin_phase,  # 1
                cos_phase,  # 1
            ),
            dim=-1,
        )

        return obs_buf, priv_obs_buf

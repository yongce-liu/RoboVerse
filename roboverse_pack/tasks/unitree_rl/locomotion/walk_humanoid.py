from __future__ import annotations

import copy
from functools import partial

import torch

from metasim.scenario.lights import DomeLightCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.scenario.simulator_params import SimParamCfg
from metasim.task.registry import register_task
from metasim.types import TensorState
from metasim.utils.math import quat_rotate_inverse
from roboverse_learn.rl.unitree_rl.configs import SensorsCfg
from roboverse_learn.rl.unitree_rl.configs.locomotion.walk_humanoid import (
    WalkHumanoidEnvCfg,
    WalkHumanoidRslRlTrainCfg,
)
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

    def _init_buffers(self):
        self.noise_scale_vec = self._get_noise_scale_vec()
        return super()._init_buffers()

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

    def _compute_task_observations(self, env_states: TensorState):
        robot_state = env_states.robots[self.robot.name]
        base_quat = robot_state.root_state[:, 3:7]
        base_lin_vel = quat_rotate_inverse(base_quat, robot_state.root_state[:, 7:10])
        base_ang_vel = quat_rotate_inverse(base_quat, robot_state.root_state[:, 10:13])
        base_euler_xyz = get_euler_xyz(base_quat)
        projected_gravity = quat_rotate_inverse(base_quat, self.gravity_vec)

        phase = self.get_phase()
        sin_phase = torch.sin(2 * torch.pi * phase).unsqueeze(1)
        cos_phase = torch.cos(2 * torch.pi * phase).unsqueeze(1)

        stance_mask = self._get_gait_phase()
        contact_mask = env_states.extras["contact_forces"][self.robot.name][:, self.feet_indices, 2] > 1.0

        q = (
            env_states.robots[self.robot.name].joint_pos - self.default_dof_pos
        ) * self.cfg.normalization.obs_scales.dof_pos
        dq = env_states.robots[self.robot.name].joint_vel * self.cfg.normalization.obs_scales.dof_vel

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
                env_states.robots[self.robot.name].joint_pos[:, self.upper_body_joint_indices]
                - self.default_dof_pos[self.upper_body_joint_indices],  # |upper_body_indices|
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


@register_task(
    "unitree_rl.walk_humanoid",
    "humanoid_walking",
    "g1.walk_humanoid",
    "walking_humanoid",
    "walkinghumanoid",
    "walk_humanoid",
)
class WalkHumanoidTask(WalkHumanoidEnv):
    """Registered humanoid locomotion task."""

    env_cfg_cls = WalkHumanoidEnvCfg
    train_cfg_cls = WalkHumanoidRslRlTrainCfg
    sensors_cls = SensorsCfg
    task_name = "humanoid_walking"

    scenario = ScenarioCfg(
        robots=["g1_dof29"],
        objects=[],
        cameras=[],
        num_envs=128,
        simulator="isaacgym",
        headless=True,
        env_spacing=2.5,
        sim_params=SimParamCfg(
            dt=0.005,
            substeps=1,
            num_threads=10,
            solver_type=1,
            num_position_iterations=4,
            num_velocity_iterations=0,
            contact_offset=0.01,
            rest_offset=0.0,
            bounce_threshold_velocity=0.5,
            max_depenetration_velocity=1.0,
            default_buffer_size_multiplier=5,
            replace_cylinder_with_capsule=True,
            friction_correlation_distance=0.025,
            friction_offset_threshold=0.04,
        ),
        lights=[
            DomeLightCfg(
                intensity=800.0,
                color=(0.85, 0.9, 1.0),
            )
        ],
    )

    def __init__(
        self,
        scenario: ScenarioCfg | None = None,
        device: str | torch.device | None = None,
        env_cfg: WalkHumanoidEnvCfg | None = None,
        sensors: SensorsCfg | dict | None = None,
    ) -> None:
        scenario_copy = copy.deepcopy(scenario or type(self).scenario)
        scenario_copy.__post_init__()

        if sensors is None:
            sensors = type(self).sensors_cls() if callable(type(self).sensors_cls) else type(self).sensors_cls

        if env_cfg is None:
            env_cfg = type(self).env_cfg_cls()

        if device is None:
            device = "cpu" if scenario_copy.simulator == "mujoco" else ("cuda" if torch.cuda.is_available() else "cpu")

        super().__init__(scenario=scenario_copy, config=env_cfg, sensors=sensors, device=device)

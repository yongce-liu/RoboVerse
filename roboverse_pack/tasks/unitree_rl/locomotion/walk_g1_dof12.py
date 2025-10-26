from __future__ import annotations

import copy

import torch

from metasim.scenario.lights import DomeLightCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.scenario.simulator_params import SimParamCfg
from metasim.task.registry import register_task
from metasim.types import TensorState
from metasim.utils.math import quat_rotate_inverse
from roboverse_learn.rl.unitree_rl.configs import SensorsCfg
from roboverse_learn.rl.unitree_rl.configs.locomotion.walk_g1_dof12 import WalkG1Dof12EnvCfg, WalkG1Dof12RslRlTrainCfg
from roboverse_pack.tasks.unitree_rl.base.base_humanoid import HumanoidTask


@register_task(
    "unitree_rl.walk_g1_dof12",
    "g1.walk_g1_dof12",
    "walk_g1_dof12",
)
class WalkG1Dof12Task(HumanoidTask):
    """Registered task wrapper with scenario defaults and cfg hooks."""

    env_cfg_cls = WalkG1Dof12EnvCfg
    train_cfg_cls = WalkG1Dof12RslRlTrainCfg
    sensors_cls = SensorsCfg
    task_name = "walk_g1_dof12"

    scenario = ScenarioCfg(
        robots=["g1_dof12"],
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
        env_cfg: WalkG1Dof12EnvCfg | None = None,
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

    def _init_buffers(self):
        self.noise_scale_vec = self._get_noise_scale_vec()
        return super()._init_buffers()

    def _get_noise_scale_vec(self):
        noise_vec = torch.zeros(size=(47,), dtype=torch.float, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.scales
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

    def _compute_task_observations(self, env_states: TensorState):
        robot_state = env_states.robots[self.name]
        base_quat = robot_state.root_state[:, 3:7]
        base_lin_vel = quat_rotate_inverse(base_quat, robot_state.root_state[:, 7:10])
        base_ang_vel = quat_rotate_inverse(base_quat, robot_state.root_state[:, 10:13])
        projected_gravity = quat_rotate_inverse(base_quat, self.gravity_vec)

        phase = self.get_phase()
        sin_phase = torch.sin(2 * torch.pi * phase).unsqueeze(1)
        cos_phase = torch.cos(2 * torch.pi * phase).unsqueeze(1)

        q = (env_states.robots[self.name].joint_pos - self.default_dof_pos) * self.cfg.normalization.obs_scales.dof_pos
        dq = env_states.robots[self.name].joint_vel * self.cfg.normalization.obs_scales.dof_vel

        obs_buf = torch.cat(
            (
                base_ang_vel * self.cfg.normalization.obs_scales.ang_vel,  # 3
                projected_gravity,  # 3
                self.commands[:, :3] * self.commands_scale,  # 3
                q,  # num_actions
                dq,  # num_actions
                self.actions,  # num_actions
                sin_phase,  # 1
                cos_phase,  # 1
            ),
            dim=-1,
        )

        # add noise if needed
        if self.add_noise:
            obs_buf += (2 * torch.rand_like(obs_buf) - 1) * self.noise_scale_vec

        priv_obs_buf = torch.cat(
            (
                base_lin_vel * self.cfg.normalization.obs_scales.lin_vel,
                base_ang_vel * self.cfg.normalization.obs_scales.ang_vel,
                projected_gravity,
                self.commands[:, :3] * self.commands_scale,
                q,  # num_actions
                dq,  # num_actions
                self.actions,
                sin_phase,
                cos_phase,
            ),
            dim=-1,
        )

        return obs_buf, priv_obs_buf

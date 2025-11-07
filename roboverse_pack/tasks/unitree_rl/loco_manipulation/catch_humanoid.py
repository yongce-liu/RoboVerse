from __future__ import annotations

import copy

import torch

from metasim.scenario.lights import DomeLightCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.scenario.simulator_params import SimParamCfg
from metasim.task.registry import register_task
from metasim.types import TensorState
from metasim.utils.math import quat_rotate_inverse
from roboverse_learn.rl.unitree_rl.configs.cfg_objects import Shuttlecock
from roboverse_learn.rl.unitree_rl.configs.loco_manipulation.catch_humanoid import (
    CatchHumanoidRslRlTrainCfg,
    CatchHumanoidTaskCfg,
)
from roboverse_pack.tasks.unitree_rl.base import LeggedRobotTask


@register_task(
    "unitree_rl.catch_g1_dof29",
    "g1.catch_g1_dof29",
    "catch_g1_dof29",
)
class CatchHumanoidTask(LeggedRobotTask):
    """Humanoid locomotion-manipulation task for catching an object."""

    env_cfg_cls = CatchHumanoidTaskCfg
    train_cfg_cls = CatchHumanoidRslRlTrainCfg
    task_name = "catch_g1_dof29"
    scenario = ScenarioCfg(
        robots=["g1_dof29"],
        objects=[Shuttlecock()],
        cameras=[],
        num_envs=128,
        simulator="isaacgym",
        headless=True,
        env_spacing=2.5,
        decimation=1,
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
        env_cfg: CatchHumanoidTaskCfg | None = None,
    ) -> None:
        scenario_copy = copy.deepcopy(scenario or type(self).scenario)
        scenario_copy.__post_init__()

        if env_cfg is None:
            env_cfg = type(self).env_cfg_cls()

        if device is None:
            device = "cpu" if scenario_copy.simulator == "mujoco" else ("cuda" if torch.cuda.is_available() else "cpu")

        super().__init__(scenario=scenario_copy, config=env_cfg, device=device)

    def _init_buffers(self):
        self.noise_scale_vec = self._get_noise_scale_vec()
        return super()._init_buffers()

    def _get_noise_scale_vec(self):
        self.num_obs_single = 3 + 3 + 3 + self.num_actions * 3
        self.num_priv_obs_single = 3 + 3 + 3 + 3 + self.num_actions * 3
        noise_vec = torch.zeros(size=(self.num_obs_single,), dtype=torch.float, device=self.device)
        self.add_noise = True
        noise_vec[:3] = 0.2  # angular velocity
        noise_vec[3:6] = 0.05  # projected gravity
        noise_vec[6:9] = 0.0  # commands
        noise_vec[9 : 9 + self.num_actions] = 0.01
        noise_vec[9 + self.num_actions : 9 + 2 * self.num_actions] = 1.5

        return noise_vec

    def _compute_task_observations(self, env_states: TensorState):
        robot_state = env_states.robots[self.name]
        base_quat = robot_state.root_state[:, 3:7]
        base_lin_vel = quat_rotate_inverse(base_quat, robot_state.root_state[:, 7:10])
        base_ang_vel = quat_rotate_inverse(base_quat, robot_state.root_state[:, 10:13])
        projected_gravity = quat_rotate_inverse(base_quat, self.gravity_vec)

        q = env_states.robots[self.name].joint_pos - self.default_dof_pos
        dq = env_states.robots[self.name].joint_vel - self.default_dof_vel

        obs_buf = torch.cat(
            (
                base_ang_vel,  # 3
                projected_gravity,  # 3
                self.commands_manager.value,  # 3
                q,  # num_actions
                dq,  # num_actions
                self.actions,  # num_actions
                # self.history_buffer['actions'][-1]  # num_actions
            ),
            dim=-1,
        )

        # add noise if needed
        if self.add_noise:
            obs_buf += (2 * torch.rand_like(obs_buf) - 1) * self.noise_scale_vec

        priv_obs_buf = torch.cat(
            (
                base_lin_vel,
                base_ang_vel,
                projected_gravity,
                self.commands_manager.value,
                q,  # num_actions
                dq,  # num_actions
                self.actions,
            ),
            dim=-1,
        )

        return obs_buf, priv_obs_buf

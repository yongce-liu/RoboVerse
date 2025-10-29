from __future__ import annotations

import copy

import torch

from metasim.scenario.lights import DomeLightCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.scenario.simulator_params import SimParamCfg
from metasim.task.registry import register_task
from metasim.types import TensorState
from metasim.utils.math import quat_rotate_inverse
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
    ) -> None:
        scenario_copy = copy.deepcopy(scenario or type(self).scenario)
        scenario_copy.__post_init__()

        if env_cfg is None:
            env_cfg = type(self).env_cfg_cls()

        if device is None:
            device = "cpu" if scenario_copy.simulator == "mujoco" else ("cuda" if torch.cuda.is_available() else "cpu")

        super().__init__(scenario=scenario_copy, config=env_cfg, device=device)

    def _init_buffers(self):
        # commands + base_ang_vel + projected_gravity + dof pos/vel/prev actions
        self.num_obs_single = 3 + 3 + 3 + self.num_actions * 3
        # commands + base_lin_vel + base_ang_vel + projected_gravity + dof pos/vel/prev actions
        self.num_priv_obs_single = 3 + 3 + 3 + 3 + self.num_actions * 3
        # Rewrite SOME Hyfer-Parameters
        self.obs_clip_limit = 100.0
        self.obs_scale = torch.ones(size=(self.num_obs_single,), dtype=torch.float, device=self.device)
        self.priv_obs_scale = torch.ones(size=(self.num_priv_obs_single,), dtype=torch.float, device=self.device)
        self.obs_noise = torch.zeros(size=(self.num_obs_single,), dtype=torch.float, device=self.device)

        ##################### for observation scale #####################
        self.obs_scale[3:6] = 0.2  # angular velocity
        self.obs_scale[9 + self.num_actions : 9 + 2 * self.num_actions] = 0.05  # joint velocity

        ##################### for priviliged observation scale #####################
        self.priv_obs_scale[6:9] = 0.2  # angular velocity
        self.priv_obs_scale[12 + self.num_actions : 12 + 2 * self.num_actions] = 0.05  # joint velocity

        ################### for noise vector ####################
        # [0:3] -> commands
        self.obs_noise[3:6] = 0.2  # [3:6] -> base_ang_vel
        self.obs_noise[6:9] = 0.05  # projected_gravity
        self.obs_noise[9 : 9 + self.num_actions] = 0.01
        self.obs_noise[9 + self.num_actions : 9 + 2 * self.num_actions] = 1.5  # joint velocities
        return super()._init_buffers()

    def _compute_task_observations(self, env_states: TensorState):
        robot_state = env_states.robots[self.name]
        base_quat = robot_state.root_state[:, 3:7]
        base_lin_vel = quat_rotate_inverse(base_quat, robot_state.root_state[:, 7:10])
        base_ang_vel = quat_rotate_inverse(base_quat, robot_state.root_state[:, 10:13])
        projected_gravity = quat_rotate_inverse(base_quat, self.gravity_vec)

        phase = self.get_phase()
        sin_phase = torch.sin(2 * torch.pi * phase).unsqueeze(1)
        cos_phase = torch.cos(2 * torch.pi * phase).unsqueeze(1)

        q = env_states.robots[self.name].joint_pos - self.default_dof_pos
        dq = env_states.robots[self.name].joint_vel

        obs_buf = torch.cat(
            (
                self.commands[:, :3],  # 3
                base_ang_vel,  # 3
                projected_gravity,  # 3
                q,  # num_actions
                dq,  # num_actions
                self.actions,  # num_actions
                sin_phase,  # 1
                cos_phase,  # 1
            ),
            dim=-1,
        )

        priv_obs_buf = torch.cat(
            (
                self.commands[:, :3],
                base_lin_vel,
                base_ang_vel,
                projected_gravity,
                q,  # num_actions
                dq,  # num_actions
                self.actions,
                sin_phase,
                cos_phase,
            ),
            dim=-1,
        )

        if self.cfg.domain_rand.add_noise2obs:
            obs_buf += (2 * torch.rand_like(obs_buf) - 1) * self.obs_noise

        # clip observations -> scale observations
        obs_buf = obs_buf.clip(-self.obs_clip_limit, self.obs_clip_limit) * self.obs_scale
        priv_obs_buf = priv_obs_buf.clip(-self.obs_clip_limit, self.obs_clip_limit) * self.priv_obs_scale

        return obs_buf, priv_obs_buf

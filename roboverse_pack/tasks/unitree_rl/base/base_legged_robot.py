from __future__ import annotations

import math
from collections import deque
from copy import deepcopy
from typing import Callable

import torch

from metasim.scenario.scenario import ScenarioCfg
from metasim.utils.dict import class_to_dict
from metasim.utils.math import quat_apply, wrap_to_pi
from metasim.utils.state import TensorState
from metasim.utils.tensor_util import torch_rand_float
from roboverse_learn.rl.unitree_rl.configs.cfg_base import BaseEnvCfg
from roboverse_learn.rl.unitree_rl.helper import get_euler_xyz, get_indices_from_substring
from roboverse_pack.robots import G1Dof12Cfg, Go2Cfg

from .base_agent import AgentTask


class LeggedRobotTask(AgentTask):
    """A base task env for legged robots."""

    def __init__(
        self,
        scenario: ScenarioCfg,
        config: BaseEnvCfg,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__(scenario=scenario, config=config, device=device)
        self.name = self.name
        self.num_actions = len(self.robot.actuators)
        self.sim_dt = self.scenario.sim_params.dt
        self.sorted_body_names = self.handler.get_body_names(self.name, sort=True)
        self.sorted_joint_names = self.handler.get_joint_names(self.name, sort=True)

        self._instantiate_cfg(self.cfg)
        self._init_rigid_body_indices()
        self._init_joint_cfg()
        self._init_reward_function()

        self._init_buffers()
        # self._init_initial_state()

    def _compute_task_observations(self, env_states: TensorState) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return (policy_obs, privileged_obs). Implemented by subclasses."""
        raise NotImplementedError

    def _instantiate_cfg(self, config: BaseEnvCfg | None):
        self.cfg = config
        # value assignments from configs
        self.decimation = self.cfg.control.decimation
        self.step_dt = self.sim_dt * self.decimation
        self.action_clip = self.cfg.control.action_clip
        self.action_scale = self.cfg.control.action_scale
        self.action_offset = self.cfg.control.action_offset
        self.common_step_counter = 0
        self.max_episode_steps = math.ceil(self.cfg.episode_length_s / self.step_dt)
        self.command_ranges = self.cfg.commands.ranges
        self.num_commands = self.cfg.commands.num_commands
        # self.reward_scales = dict(sorted(class_to_dict(self.cfg.rewards.scales).items(), key=lambda x: x[0]))
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        # self.command_ranges = class_to_dict(self.cfg.commands.ranges)

    def _init_rigid_body_indices(self):
        """Parse rigid body indices from robot cfg."""
        robot: G1Dof12Cfg | Go2Cfg = self.robot
        sorted_body_names: list[str] = self.sorted_body_names

        self.feet_indices = get_indices_from_substring(robot.feet_links, sorted_body_names).to(self.device)
        self.termination_contact_indices = get_indices_from_substring(
            robot.terminate_contacts_links, sorted_body_names
        ).to(self.device)
        self.penalised_contact_indices = get_indices_from_substring(
            robot.penalized_contacts_links, sorted_body_names
        ).to(self.device)
        # ##########################################################
        # if self.handler.scenario.simulator in ["isaacgym", "mujoco"]:
        #     self.body_ids_reindex = self.handler._get_body_ids_reindex(self.name)
        # elif self.handler.scenario.simulator == "isaacsim":
        #     sorted_body_names = self.handler.get_body_names(self.name, True)
        #     self.body_ids_reindex = torch.tensor(
        #         [self.handler.contact_sensor.body_names.index(name) for name in sorted_body_names],
        #         dtype=torch.int,
        #         device=self.handler.device,
        #     )
        # else:
        #     raise NotImplementedError(f"Simulator handler {self.handler} not supported.")
        # ################################################################

    def _init_joint_cfg(self):
        """Parse default joint positions and torque limits from cfg."""
        robot: G1Dof12Cfg | Go2Cfg = self.robot
        sorted_joint_names: list[str] = self.sorted_joint_names

        torque_limits = (
            robot.torque_limits
            if hasattr(robot, "torque_limits")
            else {name: actuator_cfg.torque_limit for name, actuator_cfg in robot.actuators.items()}
        )

        sorted_limits = [torque_limits[name] for name in sorted_joint_names]
        self.torque_limits = (
            torch.tensor(sorted_limits, device=self.device) * self.cfg.control.torque_limits_factor
        )  # (n_dof,)

        p_gains = []
        d_gains = []
        for name in sorted_joint_names:
            actuator_cfg = robot.actuators[name]
            p_gains.append(actuator_cfg.stiffness if actuator_cfg.stiffness is not None else 0.0)
            d_gains.append(actuator_cfg.damping if actuator_cfg.damping is not None else 0.0)

        self.p_gains = torch.tensor(p_gains, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.d_gains = torch.tensor(d_gains, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)

        # Check if manual PD control is needed (if any joints use effort control)
        control_types = robot.control_type
        self.manual_pd_on = any(mode == "effort" for mode in control_types.values()) if control_types else False

        dof_pos_limits = robot.joint_limits
        sorted_dof_pos_limits = [dof_pos_limits[joint] for joint in sorted_joint_names]
        self.dof_pos_limits = (
            torch.tensor(sorted_dof_pos_limits, device=self.device) * self.cfg.control.dof_pos_limits_factor
        )  # (n_dof, 2)

        # _mid = (self.dof_pos_limits[:, 0] + self.dof_pos_limits[:, 1]) / 2.0
        # _diff = self.dof_pos_limits[:, 1] - self.dof_pos_limits[:, 0]
        # soft_dof_pos_limits = torch.zeros_like(self.dof_pos_limits, device=self.device)
        # soft_dof_pos_limits[:, 0] = _mid - 0.5 * _diff * self.cfg.rewards.extras.soft_dof_pos_limit
        # soft_dof_pos_limits[:, 1] = _mid + 0.5 * _diff * self.cfg.rewards.extras.soft_dof_pos_limit
        # self.dof_pos_limits = soft_dof_pos_limits

        default_joint_pos = robot.default_joint_positions
        #######################################################
        if hasattr(self.cfg, "default_joint_positions"):
            import re

            cfg_default_joint_pos = self.cfg.default_joint_positions[robot.name]
            for joint_name, joint_pos in cfg_default_joint_pos.items():
                pattern = re.compile(joint_name)
                for name in sorted_joint_names:
                    if pattern.fullmatch(name):
                        default_joint_pos[name] = joint_pos
        #######################################################
        sorted_joint_pos = [default_joint_pos[name] for name in sorted_joint_names]
        self.default_dof_pos = torch.tensor(sorted_joint_pos, device=self.device)  # (n_dof,)

    # def _pre_physics_step(self, actions: torch.Tensor):
    #     """Apply action smoothing and wrap actions as dict before physics step."""
    #     # # low frequency action smoothing
    #     # delay = torch.rand((self.num_envs, 1), device=self.device)
    #     # actions = (1 - delay) * actions.to(self.device) + delay * self.actions
    #     # clip actions
    #     clip_action_limit = self.cfg.normalization.clip_actions
    #     actions = torch.clip(actions, -clip_action_limit, clip_action_limit).to(self.device)

    #     # TODO: add the support of multi-embodiments
    #     # should return actions_list, [List, Action:[str, RobotAction:[...]]]
    #     return actions

    def _init_reward_function(self):
        """Prepares a list of reward functions, which will be called to compute the total reward."""
        for key in list(self.reward_scales.keys()):
            if isinstance(self.reward_scales[key], tuple):
                scale, params = self.reward_scales[key][0], self.reward_scales[key][1]
            else:
                scale, params = self.reward_scales[key], {}
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] = (scale * self.step_dt, params)
        # prepare list of functions
        self.reward_functions = {}
        for name, scale in self.reward_scales.items():
            if name == "termination":
                continue
            self.reward_functions[name] = self.get_reward_fn(name, self.cfg.rewards.functions)

        # reward episode sums
        self.episode_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in self.reward_scales.keys()
        }

    def _init_buffers(self):
        self.actions = torch.zeros(
            size=(self.num_envs, self.num_actions), dtype=torch.float, device=self.device, requires_grad=False
        )
        self.rew_buf = torch.zeros(size=(self.num_envs,), dtype=torch.float, device=self.device)
        self.reset_buf = torch.zeros(size=(self.num_envs,), dtype=torch.bool, device=self.device)
        self.time_out_buf = torch.zeros(size=(self.num_envs,), dtype=torch.bool, device=self.device)

        self.up_axis_idx = 2
        self.gravity_vec = torch.tensor(
            self.get_axis_params(-1.0, self.up_axis_idx), dtype=torch.float, device=self.device
        ).repeat((self.num_envs, 1))
        self.forward_vec = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float, device=self.device).repeat((
            self.num_envs,
            1,
        ))

        self.commands = torch.zeros(
            size=(self.num_envs, self.cfg.commands.num_commands),
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        # for observation history
        env_states = self.handler.get_states()
        obs_single, priv_single = self._compute_task_observations(env_states)
        self.obs_buf_queue = deque(
            [deepcopy(obs_single) for _ in range(self.cfg.obs_len_history)], maxlen=self.cfg.obs_len_history
        )
        self.priv_obs_buf_queue = deque(
            [deepcopy(priv_single) for _ in range(self.cfg.priv_obs_len_history)], maxlen=self.cfg.priv_obs_len_history
        )

        # history buffer for reward computation
        self.history_buffer = {}
        self.history_buffer["actions"] = deque([self.actions.clone() * 0.0], maxlen=2)
        self.history_buffer["joint_vel"] = deque([self.actions.clone() * 0.0], maxlen=2)

    def _compute_effort(self, actions: torch.Tensor, env_states: TensorState) -> torch.Tensor:
        """Compute effort from actions using PD control."""
        # Scale the actions (generally output from policy)
        action_scaled = self.action_scale * actions

        # Get current joint positions and velocities
        sorted_dof_pos = env_states.robots[self.name].joint_pos
        sorted_dof_vel = env_states.robots[self.name].joint_vel

        # Compute PD control effort
        target_pos = (
            self.cfg.default_joint_pd_target if hasattr(self.cfg, "default_joint_pd_target") else self.default_dof_pos
        )
        if isinstance(target_pos, dict):
            target_pos = torch.tensor(
                [target_pos[name] for name in self.sorted_joint_names], dtype=torch.float32, device=self.device
            )
        elif not isinstance(target_pos, torch.Tensor):
            target_pos = torch.tensor(target_pos, dtype=torch.float32, device=self.device)
        target_pos = target_pos.to(self.device)
        if target_pos.dim() == 1:
            target_pos = target_pos.unsqueeze(0).repeat(self.num_envs, 1)
        if self.action_offset:
            effort = self.p_gains * (action_scaled + target_pos - sorted_dof_pos) - self.d_gains * sorted_dof_vel
        else:
            effort = self.p_gains * (action_scaled - sorted_dof_pos) - self.d_gains * sorted_dof_vel

        # Apply torque limits
        effort = torch.clip(effort, -self.torque_limits, self.torque_limits)
        return effort.to(torch.float32)

    def reset(self, env_ids: list[int] | None = None, states: TensorState | None = None):
        """Reset selected envs (defaults to all)."""
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        if len(env_ids) == 0:
            return

        if self.cfg.curriculum.enabled:
            for _name, _func in self.cfg.curriculum.funcs.items():
                _func(self, env_ids)

        # restore defaults then randomize as needed
        self._randomize_initial_state(env_ids)
        self.handler.set_states(states=self.initial_env_states, env_ids=env_ids)

        self._resample_commands(env_ids)
        for history in self.history_buffer.values():
            for item in history:
                item[env_ids] = 0.0

        self._episode_steps[env_ids] = 0
        self.actions[env_ids] = 0.0
        self.rew_buf[env_ids] = 0.0

        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["Reward/" + key] = (
                torch.mean(self.episode_sums[key][env_ids]) / self.cfg.episode_length_s
            )
            self.episode_sums[key][env_ids] = 0.0
        if self.cfg.curriculum.enabled:
            self.extras["episode"]["Curriculum/commands"] = self.command_ranges.lin_vel_x[1]
        self.extras["episode"]["Termination/time_outs"] = self.time_out_buf

    def step(
        self,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Apply actions, simulate for `decimation` steps, and compute RLTask-style outputs."""
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions, device=self.device, dtype=torch.float32)
        if actions.ndim == 1:
            actions = actions.unsqueeze(0)

        # actions = self._pre_physics_step(actions)
        self.actions[:] = actions.clip(-self.action_clip, self.action_clip).clone()
        env_states = self.get_states()
        for _ in range(self.decimation):
            if self.manual_pd_on:
                send_action = self._compute_effort(self.actions, env_states)
            else:
                send_action = self.actions * self.action_scale
            env_states = self._physics_step(send_action)

        self._post_physics_step(env_states)

        return self.obs_buf, self.rew_buf, self.reset_buf, self.time_out_buf, self.extras

    def _post_physics_step(self, env_states: TensorState):
        self._episode_steps += 1
        self.common_step_counter += 1

        self._post_physics_step_callback(env_states)
        # gym-style return values
        self.time_out_buf[:] = self._time_out(env_states)
        terminated_flags = self._terminated(env_states)
        self.reset_buf[:] = torch.logical_or(terminated_flags, self.time_out_buf)
        self.rew_buf[:] = self._reward(env_states)

        # reset envs
        # push before reset to avoid incorrect resetting of root states
        self._push_robots()

        reset_env_idx = self.reset_buf.nonzero(as_tuple=False).flatten().tolist()
        if len(reset_env_idx) > 0:
            self.reset(reset_env_idx)

        ####### Compute observations after resets ########
        # clip_obs_limit = self.cfg.normalization.clip_observations
        obs_single, priv_single = self._compute_task_observations(env_states)
        # append to the observation history buffer
        self.obs_buf_queue.append(obs_single)
        # append to the privileged observation history buffer
        if priv_single is not None and self.priv_obs_buf_queue.maxlen > 0:
            self.priv_obs_buf_queue.append(priv_single)
        ####### Compute observations after resets ########

        # copy to the history buffer
        for key, history in self.history_buffer.items():
            if hasattr(self, key):
                history.append(getattr(self, key).clone())
            elif hasattr(env_states.robots[self.name], key):
                history.append(getattr(env_states.robots[self.name], key).clone())

    def _post_physics_step_callback(self, env_states: TensorState):
        """Callback before computing terminations, rewards, and observations.

        Default behaviour: Compute ang vel command based on target and
        heading, compute measured terrain heights and randomly push robots.
        """
        env_ids = (
            (self._episode_steps % int(self.cfg.commands.resampling_time / self.step_dt) == 0)
            .nonzero(as_tuple=False)
            .flatten()
        )
        self._resample_commands(env_ids)

    def _push_robots(self):
        """Randomly set robot's root velocity to simulate a push."""
        if not self.cfg.domain_rand.push_robots:
            return
        env_states = self.handler.get_states()
        env_ids = torch.arange(self.num_envs, device=self.device)
        # push_env_ids = env_ids[
        #     torch.logical_and(
        #         self._episode_steps[env_ids] % self.cfg.domain_rand.push_interval == 0,
        #         self._episode_steps[env_ids] != 0,
        #     )
        # ]
        push_env_ids = env_ids[self._episode_steps[env_ids] % self.cfg.domain_rand.push_interval == 0,]
        if len(push_env_ids) == 0:
            return

        max_vel = self.cfg.domain_rand.max_push_vel_xy
        env_states.robots[self.name].root_state[push_env_ids, 7:9] = torch_rand_float(
            -max_vel, max_vel, (len(push_env_ids), 2), device=self.device
        )

        self.set_states(env_states, push_env_ids.tolist())

    def _reward(self, env_states):
        rew_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for name, func in self.reward_functions.items():
            scale, params = self.reward_scales[name]
            rew = scale * func(self, env_states, **params)
            rew_buf += rew
            self.episode_sums[name] += rew

        if self.cfg.rewards.only_positive_rewards:
            rew_buf[:] = torch.clip(rew_buf[:], min=0.0)

        return rew_buf

    def _terminated(self, env_states):
        contact_forces = env_states.extras["contact_forces"][self.name]
        reset_buf = torch.any(torch.norm(contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.0, dim=1)
        rpy = get_euler_xyz(env_states.robots[self.name].root_state[:, 3:7])
        reset_buf |= torch.logical_or(torch.abs(rpy[:, 1]) > 1.0, torch.abs(rpy[:, 0]) > 0.8)
        return reset_buf

    def _time_out(self, env_states: TensorState | None) -> torch.BoolTensor:
        """Timeout flags.

        Note that max_episode_steps is set to -1 by default (no timeout).
        """
        return self._episode_steps > self.max_episode_steps

    def _resample_commands(self, env_ids):
        """Randomly select commands for some environments.

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed.
        """
        if len(env_ids) == 0:
            return

        self.commands[env_ids, 0] = torch_rand_float(
            self.command_ranges.lin_vel_x[0],
            self.command_ranges.lin_vel_x[1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(
            self.command_ranges.lin_vel_y[0],
            self.command_ranges.lin_vel_y[1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(
                self.command_ranges.heading[0],
                self.command_ranges.heading[1],
                (len(env_ids), 1),
                device=self.device,
            ).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(
                self.command_ranges.ang_vel_yaw[0],
                self.command_ranges.ang_vel_yaw[1],
                (len(env_ids), 1),
                device=self.device,
            ).squeeze(1)

        # # set small commands to zero & randomly stand commands
        # low_cmd_mask = torch.norm(self.commands[env_ids, :2], dim=1) < 0.1
        random_mask = torch.rand(len(env_ids), device=self.device) < 0.02
        final_env_ids = random_mask.nonzero(as_tuple=False).flatten()
        self.commands[env_ids][final_env_ids, :] *= 0.0

        if self.cfg.commands.heading_command:
            env_states = self.get_states()
            robot_state = env_states.robots[self.name]
            base_quat = robot_state.root_state[:, 3:7]
            forward = quat_apply(base_quat, self.forward_vec)  # quat:[w, x, y, z], forward:[x, y, z]
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5 * wrap_to_pi(self.commands[:, 3] - heading), -1.0, 1.0)

    @staticmethod
    def get_reward_fn(target: str, reward_functions: list[Callable] | str) -> Callable:
        """Resolve a reward function by name from a list or module path."""
        if isinstance(reward_functions, (list, tuple)):
            fn = next((f for f in reward_functions if f.__name__ == target), None)
        elif isinstance(reward_functions, str):
            reward_module = __import__(reward_functions, fromlist=[target])
            fn = getattr(reward_module, target, None)
        else:
            raise ValueError("reward_functions should be a list of functions or a string module path")
        if fn is None:
            raise KeyError(f"No reward function named '{target}'")
        return fn

    @staticmethod
    def get_axis_params(value, axis_idx, x_value=0.0, n_dims=3):
        """Construct arguments to `Vec` according to axis index."""
        zs = torch.zeros((n_dims,))
        assert axis_idx < n_dims, "the axis dim should be within the vector dimensions"
        zs[axis_idx] = 1.0
        params = torch.where(zs == 1.0, value, zs)
        params[0] = x_value
        return params.tolist()

    def _randomize_initial_state(self, env_ids: list[int]) -> None:
        """Apply domain randomization to initial robot states."""
        if not self.cfg.domain_rand.randomize_initial_state or len(env_ids) == 0:
            return
        # weak copy/reference to avoid deepcopy
        random_initial_robot_states = self.initial_env_states.robots[self.name]
        env_tensor = torch.tensor(env_ids, dtype=torch.long, device=self.device)
        num_envs = len(env_ids)
        random_initial_robot_states.joint_pos[env_tensor] = self.default_dof_pos * torch_rand_float(
            0.5, 1.5, (num_envs, self.num_actions), device=self.device
        )
        random_initial_robot_states.joint_vel[env_tensor] = torch_rand_float(
            -1.0, 1.0, (num_envs, self.num_actions), device=self.device
        )
        random_initial_robot_states.root_state[env_tensor, :2] = torch_rand_float(
            -0.5, 0.5, (len(env_ids), 2), device=self.device
        )
        # random_yaw = torch_rand_float(-math.pi, math.pi, (len(env_ids), 1), device=self.device).flatten()
        # random_initial_robot_states.root_state[env_tensor, 3:7] = quat_from_euler_xyz(roll=random_yaw.clone()*0.0, pitch=random_yaw.clone()*0.0, yaw=random_yaw)
        # random_initial_robot_states.root_state[env_tensor, 7:13] = torch_rand_float(
        # -0.1, 0.5, (len(env_ids), 6), device=self.device
        # )
        return random_initial_robot_states

    def _build_initial_state_specs(self) -> list[dict]:
        """Return list of per-env initial states derived from config."""
        robot_state = self.cfg.initial_states.robots[self.robot.name]
        pos = robot_state.get("pos", [0.0, 0.0, 0.5])
        rot = robot_state.get("rot", [1.0, 0.0, 0.0, 0.0])
        joint_pos = robot_state.get("joint_pos", self.robot.default_joint_positions)
        joint_vel = robot_state.get("joint_vel", {name: 0.0 for name in joint_pos})

        template = {
            "objects": {},
            "robots": {
                self.robot.name: {
                    "pos": torch.tensor(pos, dtype=torch.float32),
                    "rot": torch.tensor(rot, dtype=torch.float32),
                    "dof_pos": {name: joint_pos[name] for name in joint_pos},
                    "dof_vel": {name: joint_vel[name] if name in joint_vel else 0.0 for name in joint_pos},
                }
            },
        }
        return [deepcopy(template) for _ in range(self.scenario.num_envs)]

    # def _init_initial_state(self):
    #     robot_state_cfg = self.cfg.initial_states.robots[self.name]
    #     pos = torch.tensor(robot_state_cfg.get("pos", [0.0, 0.0, 0.5]), dtype=torch.float, device=self.device)
    #     rot = torch.tensor(robot_state_cfg.get("rot", [1.0, 0.0, 0.0, 0.0]), dtype=torch.float, device=self.device)
    #     # joint_cfg = robot_state_cfg.get("joint_pos", self.robot.default_joint_positions)
    #     # joint_pos = torch.tensor(
    #     #     [joint_cfg[name] for name in self.sorted_joint_names], dtype=torch.float, device=self.device
    #     # )
    #     joint_pos = self.default_dof_pos

    #     root_state = torch.zeros(size=(self.num_envs, 13), dtype=torch.float, device=self.device)
    #     root_state[:, 0:3] = pos.unsqueeze(0).repeat(self.num_envs, 1)
    #     root_state[:, 3:7] = rot.unsqueeze(0).repeat(self.num_envs, 1)
    #     joint_pos = joint_pos.unsqueeze(0).repeat(self.num_envs, 1)

    #     body_state = torch.zeros(
    #         size=(self.num_envs, len(self.sorted_body_names), 13), dtype=torch.float, device=self.device
    #     )
    #     joint_zeros = torch.zeros_like(joint_pos)
    #     self.default_initial_state = RobotState(
    #         root_state=root_state,
    #         joint_pos=joint_pos,
    #         joint_vel=joint_zeros.clone(),
    #         body_names=self.sorted_body_names,
    #         body_state=body_state,
    #         joint_pos_target=joint_zeros.clone(),
    #         joint_vel_target=joint_zeros.clone(),
    #         joint_effort_target=joint_zeros.clone(),
    #     )

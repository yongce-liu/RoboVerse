from __future__ import annotations
from typing import Union, Callable

from collections import deque

import math
import torch

from metasim.scenario.robot import RobotCfg
from metasim.utils.state import TensorState, RobotState
from metasim.utils.tensor_util import torch_rand_float
from metasim.utils.math import quat_apply, quat_rotate_inverse, wrap_to_pi
from metasim.utils.dict import class_to_dict

from roboverse_pack.robots import G1Dof12Cfg, Go2Cfg
from roboverse_learn.unitree_rl.configs.cfg_base import BaseEnvCfg
from roboverse_learn.unitree_rl.helper import get_euler_xyz, get_indices_from_substring

from .env_base import AgentEnv, MasterSimulator


class LeggedRobotEnv(AgentEnv):
    """A base task env for legged robots."""
    def __init__(self, simulator: MasterSimulator, robot: RobotCfg, config: BaseEnvCfg) -> None:
        super().__init__(simulator, robot)
        self._instantiate_cfg(config)
        self._init_rigid_body_indices() # parse rigid body indices
        self._init_joint_cfg() # parse joint indices
        self._init_reward_function()
        self._init_buffers()
        self._init_initial_state()

    def _instantiate_cfg(self, config: BaseEnvCfg | None):
        self.cfg = config
        # value assignments from configs
        self.decimation = self.cfg.control.decimation
        self.dt = self.sim_dt * self.decimation
        self.control_type = self.cfg.control.control_type
        self.action_scale = self.cfg.control.action_scale
        self.max_episode_steps = math.ceil(self.cfg.episode_length_s / self.dt)
        self.command_ranges = self.cfg.commands.ranges
        self.num_commands = self.cfg.commands.num_commands
        self.reward_scales = dict(sorted(class_to_dict(self.cfg.rewards.scales).items(), key=lambda x: x[0]))
        # self.command_ranges = class_to_dict(self.cfg.commands.ranges)

    def _init_rigid_body_indices(self):
        """
        Parse rigid body indices from robot cfg.
        """
        robot: Union[G1Dof12Cfg, Go2Cfg] = self.robot
        sorted_body_names: list[str] = self.sorted_body_names

        self.feet_indices = get_indices_from_substring(robot.feet_links, sorted_body_names).to(self.device)
        self.termination_contact_indices = get_indices_from_substring(robot.terminate_contacts_links, sorted_body_names).to(self.device)
        self.penalised_contact_indices = get_indices_from_substring(robot.penalized_contacts_links, sorted_body_names).to(self.device)

    def _init_joint_cfg(self):
        """
        parse default joint positions and torque limits from cfg.
        """
        robot: Union[G1Dof12Cfg, Go2Cfg] = self.robot
        sorted_joint_names: list[str] = self.sorted_joint_names

        torque_limits = (
            robot.torque_limits
            if hasattr(robot, "torque_limits")
            else {name: actuator_cfg.torque_limit for name, actuator_cfg in robot.actuators.items()}
        )

        sorted_p_gains = [robot.actuators[name].stiffness for name in sorted_joint_names]
        sorted_limits = [torque_limits[name] for name in sorted_joint_names]
        self.torque_limits = torch.tensor(sorted_limits, device=self.device) * self.cfg.control.scales.torque_limits # (n_dof,)

        sorted_p_gains = [robot.actuators[name].stiffness for name in sorted_joint_names]
        self.p_gains = torch.tensor(sorted_p_gains, device=self.device) # (n_dof,)

        sorted_d_gains = [robot.actuators[name].damping for name in sorted_joint_names]
        self.d_gains = torch.tensor(sorted_d_gains, device=self.device) # (n_dof,)

        dof_pos_limits = robot.joint_limits
        sorted_dof_pos_limits = [dof_pos_limits[joint] for joint in sorted_joint_names]
        self.dof_pos_limits = torch.tensor(sorted_dof_pos_limits, device=self.device) * self.cfg.control.scales.dof_pos_limits # (n_dof, 2)

        _mid = (self.dof_pos_limits[:, 0] + self.dof_pos_limits[:, 1]) / 2.0
        _diff = self.dof_pos_limits[:, 1] - self.dof_pos_limits[:, 0]
        soft_dof_pos_limits = torch.zeros_like(self.dof_pos_limits)
        soft_dof_pos_limits[:, 0] = _mid - 0.5 * _diff * self.cfg.rewards.extras.soft_dof_pos_limit
        soft_dof_pos_limits[:, 1] = _mid + 0.5 * _diff * self.cfg.rewards.extras.soft_dof_pos_limit
        self.dof_pos_limits = soft_dof_pos_limits

        default_joint_pos = robot.default_joint_positions
        sorted_joint_pos = [default_joint_pos[name] for name in sorted_joint_names]
        self.default_dof_pos = torch.tensor(sorted_joint_pos, device=self.device) # (n_dof,)

    def _init_reward_function(self):
        """Prepares a list of reward functions, which will be called to compute the total reward."""
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name == "termination":
                continue
            self.reward_names.append(name)
            name = "reward_" + name
            self.reward_functions.append(self.get_reward_fn(name, self.cfg.rewards.functions))

        # reward episode sums
        self.episode_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in self.reward_scales.keys()
        }

    def _init_buffers(self):
        # self.joint_pos = torch.zeros(size=(self.num_envs, self.num_actions), dtype=torch.float, device=self.device, requires_grad=False)
        self.joint_vel = torch.zeros(size=(self.num_envs, self.num_actions), dtype=torch.float, device=self.device, requires_grad=False)
        self.base_pos = torch.zeros(size=(self.num_envs, 3), dtype=torch.float, device=self.device, requires_grad=False)
        self.base_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.base_euler_xyz = get_euler_xyz(self.base_quat)
        self.base_lin_vel = torch.zeros(size=(self.num_envs, 3), dtype=torch.float, device=self.device, requires_grad=False)
        self.base_ang_vel = torch.zeros(size=(self.num_envs, 3), dtype=torch.float, device=self.device, requires_grad=False)

        self.up_axis_idx = 2
        self.gravity_vec = torch.tensor(self.get_axis_params(-1.0, self.up_axis_idx), dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.contact_forces = torch.zeros(size=(self.num_envs, len(self.sorted_body_names), 3), dtype=torch.float, device=self.device)

        # self.common_step_counter = 0
        self.episode_steps = torch.zeros(size=(self.num_envs,), dtype=torch.int, device=self.device)
        self.actions = torch.zeros(size=(self.num_envs, self.num_actions), dtype=torch.float, device=self.device, requires_grad=False)
        self.torques = torch.zeros(size=(self.num_envs, self.num_actions), dtype=torch.float, device=self.device, requires_grad=False)
        self.obs_buf_history = deque([torch.zeros(size=(self.num_envs, self.cfg.num_obs_single), dtype=torch.float, device=self.device, requires_grad=False) for _ in range(self.cfg.obs_len_history)], maxlen=self.cfg.obs_len_history)
        self.obs_buf = None if self.cfg.num_obs_single == 0 else torch.cat(list(self.obs_buf_history), dim=1).to(self.device)
        self.priv_obs_buf_history = deque([torch.zeros(size=(self.num_envs, self.cfg.num_priv_obs_single), dtype=torch.float, device=self.device, requires_grad=False)], maxlen=self.cfg.priv_obs_len_history)
        self.priv_obs_buf = None if self.cfg.num_priv_obs_single == 0 else torch.cat(list(self.priv_obs_buf_history), dim=1).to(self.device)
        self.rew_buf = torch.zeros(size=(self.num_envs,), dtype=torch.float, device=self.device)
        self.reset_buf = torch.ones(size=(self.num_envs,), device=self.device, dtype=torch.bool)
        self.time_out_buf = torch.zeros(size=(self.num_envs,), device=self.device, dtype=torch.bool)
        self.extras = {}

        self.commands = torch.zeros(size=(self.num_envs, self.cfg.commands.num_commands), dtype=torch.float, device=self.device, requires_grad=False)
        self.commands_scale = torch.tensor(
            [
                self.cfg.normalization.obs_scales.lin_vel,
                self.cfg.normalization.obs_scales.lin_vel,
                self.cfg.normalization.obs_scales.ang_vel,
            ], device=self.device, requires_grad=False)

        # self.feet_air_time = torch.zeros(
        #     self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False
        # )
        # self.feet_pos = torch.zeros((self.num_envs, len(self.feet_indices), 3), device=self.device, requires_grad=False)
        # self.feet_height = torch.zeros((self.num_envs, len(self.feet_indices)), device=self.device, requires_grad=False)

        # self.rand_push_force = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        # self.rand_push_torque = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)

        # self.env_frictions = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device)
        # self.body_mass = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)

        # history buffer for reward computation
        self.history_buffer = {}
        self.history_buffer['actions'] = deque([self.actions.clone()], maxlen=1)
        self.history_buffer['joint_vel'] = deque([self.joint_vel.clone()], maxlen=1)

        # self.last_contacts = torch.zeros(
            # self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False
        # )
        # self.last_feet_z = 0.05 * torch.ones(
            # self.num_envs, len(self.feet_indices), device=self.device, requires_grad=False
        # )

    def _init_initial_state(self):
        # objects = self.cfg.initial_states.objects
        robots = self.cfg.initial_states.robots
        pos = torch.tensor(robots[self.name]["pos"], dtype=torch.float, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        rot = torch.tensor(robots[self.name]["rot"], dtype=torch.float, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        root_state = torch.zeros(size=(self.num_envs, 13), dtype=torch.float, device=self.device)
        root_state[:, 0:3] = pos
        root_state[:, 3:7] = rot
        joint_pos = torch.tensor(
            [robots[self.name]["joint_pos"][name] for name in self.sorted_joint_names],
            device=self.device,
            dtype=torch.float,
        ).unsqueeze(0).repeat(self.num_envs, 1)
        self.initial_state = RobotState(root_state=root_state,
                                     joint_pos=joint_pos,
                                     joint_vel=joint_pos.clone()*0.0,
                                     body_names=self.sorted_body_names,
                                     body_state=torch.zeros(size=(self.num_envs, len(self.sorted_body_names), 13), dtype=torch.float, device=self.device),
                                     joint_pos_target=torch.zeros(size=(self.num_envs, self.num_actions), dtype=torch.float, device=self.device),
                                     joint_vel_target=torch.zeros(size=(self.num_envs, self.num_actions), dtype=torch.float, device=self.device),
                                     joint_effort_target=torch.zeros(size=(self.num_envs, self.num_actions), dtype=torch.float, device=self.device))
        _ = self._register_initial_state(self.initial_state)

    def _compute_torques(self, env_states: TensorState, actions: torch.Tensor) -> torch.Tensor:
        dof_pos = env_states.robots[self.name].joint_pos
        dof_vel = env_states.robots[self.name].joint_vel
        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        if self.control_type=="P":
            torques = self.p_gains*(actions_scaled + self.default_dof_pos - dof_pos) - self.d_gains*dof_vel
        elif self.control_type=="V":
            torques = self.p_gains*(actions_scaled - dof_vel) - self.d_gains*(dof_vel - self.history_buffer['last_dof_vel']) / self.sim_dt
        elif self.control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def reset(self, env_ids: list[int] = None):
        if env_ids is None:
            env_ids = list(range(self.num_envs))

        if len(env_ids) == 0:
            return self.get_states()

        # randomize initial state
        # state = copy.deepcopy(self.initial_state)
        if self.cfg.domain_rand.randomize_initial_state:
            self.initial_state.joint_pos[env_ids] = self.default_dof_pos * torch_rand_float(
                0.5, 1.5, (len(env_ids), self.num_actions), device=self.device
            )
            self.initial_state.joint_vel[env_ids] = 0.0
            self.initial_state.root_state[env_ids, 7:13] = torch_rand_float(
                -0.5, 0.5, (len(env_ids), 6), device=self.device
            )
        states_to_set = self._register_initial_state(self.initial_state)
        env_states: TensorState =  super().reset(env_ids, states_to_set)

        self._resample_commands(env_ids)

        # reset state buffer in the wrapper
        for _key, _val in self.history_buffer.items():
            for _item in _val:
                _item[env_ids] = 0.0

        self.episode_steps[env_ids] = 0
        self.actions[env_ids] = 0.0
        # self.feet_air_time[env_ids] = 0.0
        # self.base_quat[env_ids] = (
        #     torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float, device=self.device)
        #     .unsqueeze(0)
        #     .repeat(len(env_ids), 1)
        # )
        # self.base_euler_xyz = get_euler_xyz(self.base_quat)
        # self.projected_gravity[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.gravity_vec[env_ids])
        self.reset_buf[env_ids] = 1

        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (torch.mean(self.episode_sums[key][env_ids]) / self.cfg.episode_length_s)
            self.episode_sums[key][env_ids] = 0.0
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.rewards.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

        # reset env handler state buffer
        for i in range(self.cfg.obs_len_history):
            self.obs_buf_history[i][env_ids] *= 0
        for i in range(self.cfg.priv_obs_len_history):
            self.priv_obs_buf_history[i][env_ids] *= 0
        return env_states

    def step(self, actions: torch.Tensor):
        clip_actions_limit = self.cfg.normalization.clip_actions
        # update self.action
        self.actions[:] = actions.clip(-clip_actions_limit, clip_actions_limit).to(self.device)
        env_states = self.get_states()
        for _ in range(self.decimation):
            self.torques[:] = self._compute_torques(env_states, self.actions)
            env_states = self._physics_step(self.torques)
        self._post_physics_step(env_states)
        return self.obs_buf, self.priv_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def _post_physics_step(self, env_states: TensorState):
        self.episode_steps += 1

        robot_state = env_states.robots[self.name]
        # update tensors from env_states
        # self.joint_pos[:] = robot_state.joint_pos
        self.joint_vel[:] = robot_state.joint_vel
        self.base_pos[:] = robot_state.root_state[:, 0:3]
        self.base_quat[:] = robot_state.root_state[:, 3:7]
        self.base_euler_xyz = get_euler_xyz(self.base_quat)
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, robot_state.root_state[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, robot_state.root_state[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.contact_forces[:] = robot_state.extra['contact_forces']

        self._post_physics_step_callback()

        # gym-style return values
        self.time_out_buf[:] = self._time_out(env_states)
        self.reset_buf[:] = torch.logical_or(self._terminated(env_states), self.time_out_buf)
        self.rew_buf[:] = self._reward(env_states)

        # reset envs
        reset_env_idx = self.reset_buf.nonzero(as_tuple=False).flatten().tolist()
        env_states = self.reset(reset_env_idx)
        # simulate the push operation
        if self.cfg.domain_rand.push_robots:
            self._push_robots(env_states)

        _tmp_obs_buf_single, _tmp_priv_obs_buf_single = self._observation(env_states)
        clip_obs_limit = self.cfg.normalization.clip_observations
        self.obs_buf_history.append(_tmp_obs_buf_single)
        self.obs_buf[:] = torch.cat(list(self.obs_buf_history), dim=1).clip(-clip_obs_limit, clip_obs_limit).to(self.device)
        if _tmp_priv_obs_buf_single is not None:
            self.priv_obs_buf_history.append(_tmp_priv_obs_buf_single)
            self.priv_obs_buf[:] = torch.cat(list(self.priv_obs_buf_history), dim=1).clip(-clip_obs_limit, clip_obs_limit).to(self.device)

        # copy to the history buffer
        for _key, _val in self.history_buffer.items():
                _val.append(self.__getattribute__(_key).clone())

    def _post_physics_step_callback(self):
        """Callback called before computing terminations, rewards, and observations
        Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        env_ids = (
            (self.episode_steps % int(self.cfg.commands.resampling_time / self.dt) == 0)
            .nonzero(as_tuple=False)
            .flatten()
        )
        self._resample_commands(env_ids)

        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)  # quat:[w, x, y, z], forward:[x, y, z]
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5 * wrap_to_pi(self.commands[:, 3] - heading), -1.0, 1.0)

    def _push_robots(self, env_states: TensorState):
        """Randomly set robot's root velocity to simulate a push."""
        env_ids = torch.arange(self.num_envs, device=self.device)
        push_env_ids = env_ids[self.episode_steps[env_ids] % self.cfg.domain_rand.push_interval == 0]
        if len(push_env_ids) == 0:
            return

        max_vel = self.cfg.domain_rand.max_push_vel_xy
        env_states.robots[self.robot.name].root_state[push_env_ids, 7:9] = torch_rand_float(
            -max_vel, max_vel, (len(push_env_ids), 2), device=self.device
        )
        # env_states.robots[self.robot.name].root_state[:, :2] += torch_rand_float(
        #     -max_vel, max_vel, (self.num_envs, 2), device=self.device
        # )*self.dt

        # max_angular = self.cfg.random.push.max_push_ang_vel
        # env_states.robots[self.robot.name].root_state[:, 10:13] = torch_rand_float(
        #     -max_angular, max_angular, (self.num_envs, 3), device=self.device
        # )
        self.set_states(env_states, push_env_ids.tolist())

    def _reward(self, env_states):
        rew_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            unscaled_rew = self.reward_functions[i](self)
            rew = unscaled_rew * self.reward_scales[name]
            rew_buf += rew
            self.episode_sums[name] += rew

        if self.cfg.rewards.only_positive_rewards:
            rew_buf[:] = torch.clip(rew_buf[:], min=0.0)

        return rew_buf

    def _terminated(self, env_states):
        # contact_forces = env_states.robots[self.name].extra["contact_forces"]
        reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.0, dim=1)
        reset_buf |= torch.logical_or(torch.abs(self.base_euler_xyz[:, 1]) > 1.0, torch.abs(self.base_euler_xyz[:, 0]) > 0.8)
        return reset_buf

    def _time_out(self, env_states: TensorState | None) -> torch.BoolTensor:
        """
        Timeout flags.
        Note that max_episode_steps is set to -1 by default (no timeout).
        """
        return self.episode_steps > self.max_episode_steps

    def _resample_commands(self, env_ids):
        """Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
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

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    @staticmethod
    def get_reward_fn(target: str, reward_functions: list[Callable] | str) -> Callable:
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
        """construct arguments to `Vec` according to axis index."""
        zs = torch.zeros((n_dims,))
        assert axis_idx < n_dims, "the axis dim should be within the vector dimensions"
        zs[axis_idx] = 1.0
        params = torch.where(zs == 1.0, value, zs)
        params[0] = x_value
        return params.tolist()

    @property
    def num_obs(self) -> int:
        return int(self.cfg.num_obs_single * self.cfg.obs_len_history)

    @property
    def num_priv_obs(self) -> int:
        return int(self.cfg.num_priv_obs_single * self.cfg.priv_obs_len_history)

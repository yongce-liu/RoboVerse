from __future__ import annotations
import os
import torch
from loguru import logger as log
import mujoco
import mujoco.viewer
from metasim.cfg.scenario import ScenarioCfg
import yaml
import numpy as np
from metasim.utils.state import list_state_to_tensor
import rootutils
from metasim.utils.humanoid_robot_util import (
    get_euler_xyz_tensor,
)

PROJECT_ROOT_DIR = str(rootutils.setup_root(__file__, pythonpath=True))
UNITREE_GYM_ROOT_DIR = os.path.join(PROJECT_ROOT_DIR, 'roboverse_learn', 'unitree_rl')

class IndependentMujocoController:
    def __init__(self, args, scenario: ScenarioCfg):
        self.device = torch.device("cuda" if torch.cuda.is_available else "cpu")
        config_file = args.config_file
        with open(f"{UNITREE_GYM_ROOT_DIR}/deploy/configs/mujoco/{config_file}", "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            policy_path = config["policy_path"].replace("{PROJECT_ROOT_DIR}", PROJECT_ROOT_DIR)
            xml_path = config["xml_path"].replace("{PROJECT_ROOT_DIR}", PROJECT_ROOT_DIR)

            simulation_duration = config["simulation_duration"]
            simulation_dt = config["simulation_dt"]
            self.control_decimation = config["control_decimation"]

            kps = np.array(config["kps"], dtype=np.float32)
            kds = np.array(config["kds"], dtype=np.float32)

            default_angles = np.array(config["default_angles"], dtype=np.float32)

            ang_vel_scale = config["ang_vel_scale"]
            dof_pos_scale = config["dof_pos_scale"]
            dof_vel_scale = config["dof_vel_scale"]
            action_scale = config["action_scale"]
            cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

            num_actions = config["num_actions"]
            num_obs = config["num_obs"]

            cmd = np.array(config["cmd_init"], dtype=np.float32)

        self.episode_length_buf = torch.ones(1, dtype=torch.int32)
        self.policy = torch.jit.load(policy_path)
        self.policy.to(self.device)
        self.physics = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.physics)
        mujoco.mj_resetDataKeyframe(self.physics, self.data, 0)
        joint_names = []
        for i in range(self.physics.nu):
            name = mujoco.mj_id2name(self.physics, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            joint_names.append(name)
        self.joint_names = joint_names
        self.sorted_joint_names = self.joint_names.copy()
        self.sorted_joint_names.sort()
        self.joint_name2idx = {name: idx for idx, name in enumerate(joint_names)}
        self.joint_reindex_cache = [self.joint_names.index(jn) for jn in self.sorted_joint_names]
        self.joint_reindex_cache_inverse = [self.sorted_joint_names.index(jn) for jn in self.joint_names]


        self.physics.opt.timestep = simulation_dt
        self._parse_cfg(scenario)
        self._init_buffers()
        self._parse_joint_cfg(scenario)
        self._init_torque_control()

    @staticmethod
    def get_axis_params(value, axis_idx, x_value=0.0, n_dims=3):
        """construct arguments to `Vec` according to axis index."""
        zs = torch.zeros((n_dims,))
        assert axis_idx < n_dims, "the axis dim should be within the vector dimensions"
        zs[axis_idx] = 1.0
        params = torch.where(zs == 1.0, value, zs)
        params[0] = x_value
        return params.tolist()

    def _init_buffers(self):
        """
        Init all buffer for reward computation
        """
        self.up_axis_idx = 2
        self.base_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)
        self.actions = torch.zeros(self.num_envs, self.num_actions, device=self.device)
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=torch.float)
        self.gravity_vec = torch.tensor(
            self.get_axis_params(-1.0, self.up_axis_idx), device=self.device, dtype=torch.float32
        ).repeat((
            self.num_envs,
            1,
        ))
        self.forward_vec = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=self.device).repeat((
            self.num_envs,
            1,
        ))
        self.common_step_counter = 0
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.int)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.commands = torch.zeros(
            self.num_envs,
            self.cfg.commands.num_commands,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.extras = {}
        self.commands_scale = torch.tensor(
            [
                self.cfg.normalization.obs_scales.lin_vel,
                self.cfg.normalization.obs_scales.lin_vel,
                self.cfg.normalization.obs_scales.ang_vel,
            ],
            device=self.device,
            requires_grad=False,
        )
        self.base_lin_vel = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.base_ang_vel = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)

    def _parse_cfg(self, scenario: ScenarioCfg):
        # loading task-specific configuration
        self.dt = scenario.decimation * scenario.sim_params.dt
        self.command_ranges = scenario.task.command_ranges
        self.num_commands = scenario.task.command_dim

        self.scenario = scenario
        self.robot = scenario.robots[0]
        self.num_envs = scenario.num_envs
        self.num_obs = scenario.task.num_observations
        self.num_actions = scenario.task.num_actions
        self.num_privileged_obs = scenario.task.num_privileged_obs
        self.max_episode_length = scenario.task.max_episode_length
        self._action_scale = scenario.control.action_scale
        self._action_offset = scenario.control.action_offset
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cfg = scenario.task
        from metasim.utils.dict import class_to_dict
        self.train_cfg = class_to_dict(scenario.task.ppo_cfg)

    def _parse_joint_cfg(self, scenario):
        """
        parse default joint positions and torque limits from cfg.
        """

        torque_limits = scenario.robots[0].torque_limits
        sorted_joint_names = sorted(torque_limits.keys())
        sorted_limits = [torque_limits[name] for name in sorted_joint_names]
        self.cfg.torque_limits = (
            torch.tensor(sorted_limits, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
            * scenario.control.torque_limit_scale
        )

        default_joint_pos = scenario.robots[0].default_joint_positions
        sorted_joint_pos = [default_joint_pos[name] for name in sorted_joint_names]
        self.cfg.default_joint_pd_target = (
            torch.tensor(sorted_joint_pos).unsqueeze(0).repeat(self.num_envs, 1).to(self.device)
        )

    def _init_torque_control(self):
        """Initialize torque control parameters based on robot configuration."""
        joint_names = self.joint_names
        self._robot_num_dof = len(joint_names)
        self._effort_controlled_joints = []
        self._p_gains = np.zeros(self._robot_num_dof)
        self._d_gains = np.zeros(self._robot_num_dof)
        self._torque_limits = np.zeros(self._robot_num_dof)

        self._manual_pd_on = any(mode == "effort" for mode in self.robot.control_type.values())

        default_dof_pos = []

        for i, joint_name in enumerate(joint_names):
            i_actuator_cfg = self.robot.actuators[joint_name]
            i_control_mode = self.robot.control_type.get(joint_name, "position")

            if joint_name in self.robot.default_joint_positions:
                default_pos = self.robot.default_joint_positions[joint_name]
            else:
                joint_id = self.physics.joint(joint_name).id
                joint_range = self.physics.jnt_range[joint_id]
                default_pos = 0.3 * (joint_range[0] + joint_range[1])
            default_dof_pos.append(default_pos)

            if i_control_mode == "effort":
                self._effort_controlled_joints.append(i)
                self._p_gains[i] = i_actuator_cfg.stiffness
                self._d_gains[i] = i_actuator_cfg.damping

                if i_actuator_cfg.torque_limit is not None:
                    torque_limit = i_actuator_cfg.torque_limit
                else:
                    torque_limit = self.robot.torque_limits[joint_name]

                self._torque_limits[i] = self.scenario.control.torque_limit_scale * torque_limit

            elif i_control_mode == "position":
                self._position_controlled_joints.append(i)
            else:
                log.error(f"Unknown actuator control mode: {i_control_mode}, only support effort and position")
                raise ValueError

        self._robot_default_dof_pos = np.array(default_dof_pos)
        self._current_vel_target = None  # Initialize velocity target tracking

    def _get_phase(self):
        cycle_time = self.cfg.reward_cfg.cycle_time
        phase = self.episode_length_buf * self.dt / cycle_time
        return phase

    def compute_observations(self):
        """compute observation and privileged observation."""

        phase = self._get_phase()

        sin_pos = torch.sin(2 * torch.pi * phase).unsqueeze(1)
        cos_pos = torch.cos(2 * torch.pi * phase).unsqueeze(1)
        self.command_input = torch.cat((sin_pos, cos_pos, self.commands[:, :3] * self.commands_scale), dim=1)
        self.command_input_wo_clock = self.commands[:, :3] * self.commands_scale
        reindex = self.joint_reindex_cache
        inverse_index = self.joint_reindex_cache_inverse
        q = (torch.from_numpy(self.data.qpos[7:]).to(self.device).unsqueeze(0)[:, reindex] - self.cfg.default_joint_pd_target)# * self.cfg.normalization.obs_scales.dof_pos
        dq = torch.from_numpy(self.data.qvel[6:]).to(self.device).unsqueeze(0)# * self.cfg.normalization.obs_scales.dof_vel
        ##TODO: add _update_odometry_tensors
        obs = torch.cat(
            (
                torch.zeros_like(self.command_input_wo_clock),  # 3
                q,  # |A|
                torch.zeros_like(dq),  # |A|
                torch.zeros_like(self.actions),
                torch.zeros_like(self.base_ang_vel * self.cfg.normalization.obs_scales.ang_vel),  # 3
                torch.zeros_like(self.base_euler_xyz * self.cfg.normalization.obs_scales.quat),  # 3
            ),
            dim=-1,
        ).to(torch.float32)
        self.obs_buf = obs

    def set_dof_targets(self, actions) -> None:
        self._actions_cache = actions
        reverse_reindex = self.joint_reindex_cache_inverse
        reindex = self.joint_reindex_cache
        joint_targets = actions
        if self._manual_pd_on:

            self._current_action = np.zeros(self._robot_num_dof)
            self._current_action = np.array(joint_targets)
            efforts = self._compute_effort(joint_targets)
            for i, joint in enumerate(self.joint_names):
                index = self.joint_name2idx[joint]
                self.data.ctrl[index] = efforts[0, i]
            # self.data.ctrl = efforts
            # for i in self._position_controlled_joints:
            #     joint_name = joint_names[i]
            #     if joint_name in joint_targets:
            #         actuator = self.physics.data.actuator(f"{self._mujoco_robot_name}{joint_name}")
            #         actuator.ctrl = joint_targets[joint_name]
        else:
            # joint_targets = actions[0][obj_name]["dof_pos_target"]
            # for joint_name, target_pos in joint_targets.items():
            #     actuator = self.physics.data.actuator(f"{self._mujoco_robot_name}{joint_name}")
            #     actuator.ctrl = target_pos
            self._current_action = np.array(actions)
            self.data.ctrl = joint_targets

    def _compute_effort(self, actions):
        """Compute effort from actions using PD controller."""
        action_scaled = self._action_scale * actions
        robot_dof_pos = np.array(self.data.qpos[7:])
        robot_dof_vel = np.array(self.data.qvel[6:])

        if self._action_offset:
            effort = (
                    self._p_gains * (action_scaled + self._robot_default_dof_pos - robot_dof_pos)
                    - self._d_gains * robot_dof_vel
            )
        else:
            effort = self._p_gains * (action_scaled - robot_dof_pos) - self._d_gains * robot_dof_vel
        # effort *= 0.5
        effort = np.clip(effort, -self._torque_limits, self._torque_limits)

        return effort

    def step(self, counter):
        self.compute_observations()
        if counter % self.control_decimation == 0:
            actions = self.policy(self.obs_buf).detach()
            delay = torch.rand((self.num_envs, 1), device=self.device)
            self.actions = (1 - delay) * actions + delay * self.actions
            # self.actions = torch.zeros_like(self.actions)
            self.set_dof_targets(self.actions.cpu().detach().numpy())
        mujoco.mj_step(self.physics, self.data)

    def launch(self):
        count = 0
        with mujoco.viewer.launch_passive(self.physics, self.data) as viewer:
            while viewer.is_running():
                self.step(count)
                viewer.sync()
                count += 1

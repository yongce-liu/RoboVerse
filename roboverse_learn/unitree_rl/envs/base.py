from __future__ import annotations

import copy

import torch
from loguru import logger as log
from rsl_rl.env import VecEnv

from metasim.cfg.scenario import ScenarioCfg
from metasim.constants import SimType
from metasim.sim.env_wrapper import EnvWrapper
from metasim.utils.setup_util import get_sim_env_class


class BaseEnv(VecEnv):
    """
    Wraps Metasim environments to be compatible with rsl_rl OnPolicyRunner.

    Note that rsl_rl is designed for parallel training fully on GPU, with robust support for Isaac Gym and Isaac Lab.
    """

    def __init__(self, scenario: ScenarioCfg):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu").type
        if SimType(scenario.sim) in [SimType.ISAACGYM, SimType.ISAACLAB, SimType.GENESIS, SimType.MJX]:
            log.info(f"RslRlWrapper uses {SimType(scenario.sim)} simulator.")
        elif SimType(scenario.sim) in [SimType.MUJOCO]:
            assert scenario.num_envs == 1, "MuJoCo only supports single environment in rsl_rl wrapper."
            self.device = torch.device("cpu").type
            log.warning(f"Only for simulation, not for training, using {SimType(scenario.sim)} simulator.")
        else:
            raise NotImplementedError(
                f"RslRlWrapper in Roboverse now only supports {SimType.ISAACGYM}, but got {scenario.sim}"
            )

        # TODO read camera config
        # self.env.cfg.sensor.camera

        # load simulator handler
        env_class = get_sim_env_class(SimType(scenario.sim))
        self.env: EnvWrapper = env_class(scenario)
        self._parse_cfg(scenario)
        self._get_init_states(scenario)

    def _parse_cfg(self, scenario: ScenarioCfg):
        # loading task-specific configuration
        self.scenario = scenario
        self.robot = scenario.robots[0]
        self.robots = scenario.robots
        self.num_envs = scenario.num_envs
        self.num_obs = scenario.task.num_observations
        self.num_actions = scenario.task.num_actions
        self.num_privileged_obs = scenario.task.num_privileged_obs
        # self.max_episode_length = scenario.task.max_episode_length
        self.max_episode_length = scenario.task.episode_length
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cfg = scenario.task
        from metasim.utils.dict import class_to_dict

        self.train_cfg = class_to_dict(scenario.task.ppo_cfg)
        self.object_names = sorted({obj.name for obj in scenario.objects})
        self.robot_names = sorted({robot.name for robot in scenario.robots})

    def _get_init_states(self, scenario):
        """Get initial states from the scenario configuration."""
        init_states_list = getattr(scenario.task, "init_states", None)
        if init_states_list is None:
            raise AttributeError("'task cfg' has no attribute 'init_states', please add it in your scenario config!")
        init_states_list = [
            {
                "objects": {key: es["objects"][key] for key in es["objects"] if key in self.object_names},
                "robots": {key: es["robots"][key] for key in es["robots"] if key in self.robot_names},
            }
            for es in init_states_list
        ]

        if len(init_states_list) < self.num_envs:
            copies_needed = self.num_envs // len(init_states_list)
            remainder = self.num_envs % len(init_states_list)

            init_states_list = [copy.deepcopy(state) for state in init_states_list for _ in range(copies_needed)] + [
                copy.deepcopy(state) for state in init_states_list[:remainder]
            ]
        else:
            init_states_list = init_states_list[: self.num_envs]

        self.init_states = init_states_list

    def get_observations(self):
        """design from config"""
        return self.obs_buf

    def get_privileged_observations(self):
        """design from config"""
        return self.privileged_obs_buf

    def reset(self):
        """Reset all robots"""
        self.reset_idx(list(range(self.num_envs)))
        obs, privileged_obs, _, _, _ = self.step(
            torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False)
        )
        return obs, privileged_obs

    def reset_idx(self, env_ids=None):
        raise NotImplementedError

    def step(self, actions):
        raise NotImplementedError

    def compute_observations(self):
        raise NotImplementedError

"""Rsl_rl 1.0.2 wrapper, align OnPolicyRunnner in rsl_rl with metasim."""

from __future__ import annotations

import torch
from loguru import logger as log
from rsl_rl.env import VecEnv

from metasim.cfg.scenario import ScenarioCfg
from metasim.constants import SimType
from metasim.sim.env_wrapper import EnvWrapper
from metasim.utils.setup_util import get_sim_env_class

from metasim.utils.state import list_state_to_tensor



class RslRlWrapper(VecEnv):
    """
    Wraps Metasim environments to be compatible with rsl_rl OnPolicyRunner.

    Note that rsl_rl is designed for parallel training fully on GPU, with robust support for Isaac Gym and Isaac Lab.
    """

    def __init__(self, scenario: ScenarioCfg):
        super().__init__()


        # if SimType(scenario.sim) not in [SimType.ISAACGYM,SimType.ISAACLAB, SimType.GENESIS]:
        #     raise NotImplementedError(
        #         f"RslRlWrapper in Roboverse now only supports {SimType.ISAACGYM}, but got {scenario.sim}"
        #     )
        if SimType(scenario.sim) in [SimType.MUJOCO]:
            assert scenario.num_envs == 1, "RslRlWrapper only supports single env for MuJoCo, please set num_envs=1 in your scenario config!"
        self.device = torch.device("cuda" if torch.cuda.is_available else "cpu")
        log.info(f"using device {self.device}")

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
        self.num_envs = scenario.num_envs
        self.num_obs = scenario.task.num_observations
        self.num_actions = scenario.task.num_actions
        self.num_privileged_obs = scenario.task.num_privileged_obs
        self.max_episode_length = scenario.task.max_episode_length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cfg = scenario.task
        from metasim.utils.dict import class_to_dict
        self.train_cfg = class_to_dict(scenario.task.ppo_cfg)


    def _get_init_states(self, scenario):
        """ Get initial states from the scenario configuration."""

        init_states_list = getattr(scenario.task, 'init_states', None)
        if init_states_list is None:
            raise AttributeError(f"'task cfg' has no attribute 'init_states', please add it in your scenario config!")

        if len(init_states_list) < self.num_envs:
            init_states_list = (
                init_states_list * (self.num_envs // len(init_states_list))
                + init_states_list[: self.num_envs % len(init_states_list)]
            )
        else:
            init_states_list = init_states_list[: self.num_envs]

        self.init_states = init_states_list

        if scenario.sim == SimType.ISAACGYM:
            #tensorize the initial states as TensorState, now we only support IsaacGym
            self.init_states = list_state_to_tensor(self.env.handler, init_states_list, device=self.device)


    def get_observations(self):
        """design from config"""
        return self.obs_buf

    def get_privileged_observations(self):
        """design from config"""
        return self.privileged_obs_buf

    def step(self, actions):
        raise NotImplementedError

    def get_visual_observations(self):
        raise NotImplementedError

    def reset(self, env_ids=None):
        """
        Reset state in the env and buffer in the wrapper
        """
        if env_ids is None:
            env_ids = list(range(self.handler.num_envs))

        # reset in the env
        self.env.reset(env_ids)

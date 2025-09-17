from __future__ import annotations
from typing import Union
import torch

from roboverse_learn.rl.unitree_rl.envs import EnvTypes
from roboverse_learn.rl.unitree_rl.configs.cfg_base import RslRlTrainCfg
from .master import BaseRunnerWrapper


class RslRlEnvWrapper:
    def __init__(self, env):
        self.env = env

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, Union[torch.Tensor, None], torch.Tensor, torch.Tensor, dict]:
        _ = self.env.step(actions)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def reset(self, env_ids: Union[list, torch.Tensor] = None):
        _ = self.env.reset(list(range(self.num_envs)))
        _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return self.obs_buf, self.privileged_obs_buf

    def get_observations(self) -> torch.Tensor:
        return self.obs_buf

    def get_privileged_observations(self) -> Union[torch.Tensor, None]:
        return self.privileged_obs_buf

    @property
    def num_envs(self):
        return self.env.num_envs

    @property
    def num_obs(self):
        return self.env.num_obs

    @property
    def num_privileged_obs(self):
        return self.env.num_priv_obs

    @property
    def num_actions(self):
        return self.env.num_actions

    @property
    def max_episode_length(self):
        return self.env.max_episode_steps

    @property
    def privileged_obs_buf(self):
        return self.env.priv_obs_buf

    @property
    def obs_buf(self):
        return self.env.obs_buf

    @property
    def rew_buf(self):
        return self.env.rew_buf

    @property
    def reset_buf(self):
        return self.env.reset_buf

    @property
    def episode_length_buf(self):
        return self.env.episode_steps

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        self.env.episode_steps = value

    @property
    def extras(self):
        return self.env.extras

    @property
    def device(self):
        return self.env.device


class RslRlWrapper(BaseRunnerWrapper):
    def __init__(self, env: EnvTypes, train_cfg: dict|RslRlTrainCfg, log_dir:str):
        super().__init__(env, train_cfg, log_dir)
        from rsl_rl.runners.on_policy_runner import OnPolicyRunner

        self.env_wrapper = RslRlEnvWrapper(self.env)
        self.runner = OnPolicyRunner(
            env=self.env_wrapper,
            train_cfg=self.train_cfg,
            device=self.device,
            log_dir=log_dir,
        )

    def learn(self, max_iterations=10000):
        self.runner.learn(num_learning_iterations=max_iterations, init_at_random_ep_len=True)

    def load(self, path):
        self.runner.load(path)

    def get_policy(self):
        return self.runner.get_inference_policy()

from __future__ import annotations
from typing import Union
import torch

from roboverse_learn.unitree_rl.configs.cfg_base import RslRlTrainCfg
from .base import EnvTypes, BaseWrapper


class RslRLEnvWrapper:
    def __init__(self, env: EnvTypes):
        self.env = env

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, Union[torch.Tensor, None], torch.Tensor, torch.Tensor, dict]:
        _ = self.env.step(actions)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def reset(self, env_ids: Union[list, torch.Tensor] = None):
        _ = self.env.reset(env_ids)
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


class RslRlWrapper(BaseWrapper):
    def __init__(self, env: EnvTypes, train_cfg: dict|RslRlTrainCfg, log_dir:str):
        from rsl_rl.runners.on_policy_runner import OnPolicyRunner
        self.env = RslRLEnvWrapper(env)
        self.device = env.device
        if not isinstance(train_cfg, dict):
            train_cfg = train_cfg.to_dict()
        self.train_cfg = train_cfg
        self.log_dir = log_dir

        self.runner = OnPolicyRunner(
            env=self.env,
            train_cfg=self.train_cfg,
            device=self.device,
            log_dir=log_dir,
        )

    def learn(self, max_iterations=10000):
        self.runner.learn(num_learning_iterations=max_iterations, init_at_random_ep_len=True)

    def load(self, path):
        self.runner.load(path)

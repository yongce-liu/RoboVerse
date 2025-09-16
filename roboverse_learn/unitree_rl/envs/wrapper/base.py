from __future__ import annotations
from typing import Union
from ..env_base import AgentEnv
from ..env_legged_robot import LeggedRobotEnv
from ..env_humanoid import HumanoidEnv


EnvTypes = Union[AgentEnv, LeggedRobotEnv, HumanoidEnv]
class BaseWrapper:
    def __init__(self, env: EnvTypes, train_cfg: dict, log_dir: str):
        raise NotImplementedError

    def load(self, path):
        raise NotImplementedError

    def learn(self, max_iterations):
        raise NotImplementedError

# ruff: noqa: F401
"""Environment base classes and type aliases for Unitree RL tasks."""

from typing import Union

from .env_base import AgentEnv, MasterSimulator
from .env_humanoid import HumanoidEnv
from .env_legged_robot import LeggedRobotEnv

EnvTypes = Union[AgentEnv, LeggedRobotEnv, HumanoidEnv]

"""Environment base classes and type aliases for Unitree RL tasks."""

from typing import Union

from .base_agent import AgentTask
from .base_humanoid import HumanoidTask
from .base_legged_robot import LeggedRobotTask

EnvTypes = Union[AgentTask, LeggedRobotTask, HumanoidTask]

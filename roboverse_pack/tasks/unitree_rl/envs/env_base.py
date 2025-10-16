from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import asdict

import torch

from metasim.constants import SimType
from metasim.scenario.robot import RobotCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.sim.base import BaseSimHandler
from metasim.types import Action, Info, Reward, RobotState, Success, TensorState, TimeOut
from metasim.utils.setup_util import get_sim_handler_class
from roboverse_learn.rl.unitree_rl.configs import SensorsCfg


class MasterSimulator:
    """Top-level simulator wrapper that manages multiple environments.

    Provides a unified interface to instantiate the simulator backend,
    initialize states, and step physics.
    """

    def __init__(
        self,
        scenario: ScenarioCfg | None = None,
        sensors: SensorsCfg | dict | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        """Initialize the task env.

        Args:
            scenario: The scenario configuration. If None, it will use the class variable "scenario".
            sensors: Optional sensor configuration or dict passed to the handler.
            device: The device to use for the environment. If None, it will use "cuda" if available, otherwise "cpu".
        """
        self.sensors: dict = sensors if isinstance(sensors, (dict, type(None))) else asdict(sensors)
        self.device: str = str(device) if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self._instantiate_env(scenario)
        self._initialize_states()
        assert isinstance(self.initial_states, TensorState), "initial_states should be of type TensorState"

    def _instantiate_env(self, scenario: ScenarioCfg) -> None:
        """Instantiate the environment.

        Args:
            scenario: The scenario configuration
        """
        # weak copy to self space
        self.scenario = scenario
        self.robots = scenario.robots
        self.objects = scenario.objects
        # value assignments
        self.num_envs = self.scenario.num_envs
        # handlers & start
        handler_class = get_sim_handler_class(SimType(scenario.simulator))
        self.handler: BaseSimHandler = handler_class(scenario, self.optional_queries)
        self.handler.launch()

    def _initialize_states(self) -> None:
        self.initial_states: TensorState = deepcopy(self.handler.get_states(mode="tensor"))
        for obj in self.objects:
            if hasattr(obj, "root_state"):
                self.initial_states.objects[obj.name].root_state[:, :] = (
                    torch.tensor(obj.root_state, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
                )

    def _physics_step(self, actions: Action) -> TensorState:
        """Physics step callback."""
        self.handler.set_dof_targets(actions)
        self.handler.simulate()
        env_states = self.handler.get_states()
        return env_states

    def close(self) -> None:
        """Close the environment."""
        self.handler.close()

    @property
    def optional_queries(self):
        """Optional sensor queries passed into the simulator handler."""
        return self.sensors


class AgentEnv:
    """A base sub env for each embodiment in the env."""

    def __init__(self, simulator: MasterSimulator, robot: RobotCfg) -> None:
        """In this environment, we share the env_states for all robots, you can choose to do all/partial/no other robots' obs."""
        self._copy_shared_values(simulator, robot)
        self.max_episode_steps = -1  # to be set by task env
        self.episode_steps = torch.zeros(size=(self.num_envs,), dtype=torch.int, device=self.device)
        self.actions = torch.zeros(
            size=(self.num_envs, self.num_actions), dtype=torch.float, device=self.device, requires_grad=False
        )
        self.torques = torch.zeros(
            size=(self.num_envs, self.num_actions), dtype=torch.float, device=self.device, requires_grad=False
        )
        self.obs_buf_queue: deque = deque(maxlen=0)
        self.priv_obs_buf_queue: deque = deque(maxlen=0)
        self.rew_buf = torch.zeros(size=(self.num_envs,), dtype=torch.float, device=self.device)
        self.reset_buf = torch.ones(size=(self.num_envs,), device=self.device, dtype=torch.bool)
        self.time_out_buf = torch.zeros(size=(self.num_envs,), device=self.device, dtype=torch.bool)
        self.extras = {}

    def _copy_shared_values(self, simulator: MasterSimulator, robot: RobotCfg) -> None:
        # for simulator values
        self.simulator = simulator
        self.device = self.simulator.device
        self.num_envs = self.simulator.num_envs
        self.sim_dt = self.simulator.scenario.sim_params.dt
        # for robot values
        self.robot = robot
        self.name = robot.name
        self.num_actions = len(self.robot.actuators)
        self.sorted_body_names = self.simulator.handler.get_body_names(self.name, sort=True)
        self.sorted_joint_names = self.simulator.handler.get_joint_names(self.name, sort=True)

    def _register_initial_state(self, init_state: RobotState) -> None:
        # self.simulator.initial_states.robots[self.name]
        for key, value in vars(init_state).items():
            if value is not None:
                setattr(self.simulator.initial_states.robots[self.name], key, value)
            # else:
            #     default_value = getattr(self.simulator.initial_states.robots[self.name], key)
            #     setattr(self.simulator.initial_states.robots[self.name], key, default_value*0.0)
        return self.simulator.initial_states

    def get_states(self) -> TensorState:
        """Get the current state of the environment."""
        return self.simulator.handler.get_states()

    def set_states(self, states: TensorState, env_ids: list[int] | None = None) -> None:
        """Set the state of the environment.

        Args:
            states: The states to set.
            env_ids: The environment ids to set. If None, set all environments.
        """
        self.simulator.handler.set_states(states=states, env_ids=env_ids)

    def reset(self, env_ids: list[int], states: TensorState | None = None) -> TensorState:
        """Reset the environment.

        Args:
            env_ids: The environment ids to reset
            states: Optional external states to set for the selected envs. If None, use initial states.

        Returns:
            TensorState: The TensorStateervation
            priv_TensorState: The privileged TensorStateervation
            info: The info
        """
        states_to_set = self.simulator.initial_states if states is None else states
        self.simulator.handler.set_states(states=states_to_set, env_ids=env_ids)
        env_states = self.simulator.handler.get_states(env_ids=env_ids)

        return env_states

    def step(self, actions: Action) -> tuple[TensorState, Reward, Success, TimeOut, Info | None]:
        """Step the environment with the given action(s)."""
        raise NotImplementedError

    def _physics_step(self, actions) -> TensorState:
        return self.simulator._physics_step(actions)

    def _observation(self, env_states: TensorState) -> torch.Tensor:
        """Get the Observation & Privileged Observation of the environment."""
        raise NotImplementedError

    def _reward(self, env_states: TensorState) -> Reward:
        """Get the reward of the environment."""
        raise NotImplementedError

    def _terminated(self, env_states: TensorState) -> torch.BoolTensor:
        """Get the terminated of the environment."""
        raise NotImplementedError

    def _time_out(self, env_states: TensorState | None) -> torch.BoolTensor:
        raise NotImplementedError

    @property
    def obs_buf(self):
        """Stacked observation buffer with history along features."""
        return torch.cat(list(self.obs_buf_queue), dim=1)

    @property
    def priv_obs_buf(self):
        """Stacked privileged observation buffer with history along features."""
        return torch.cat(list(self.priv_obs_buf_queue), dim=1)

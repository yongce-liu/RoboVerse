from __future__ import annotations

import torch

from metasim.constants import SimType
from metasim.scenario.scenario import ScenarioCfg
from metasim.sim.base import BaseSimHandler
from metasim.types import Action, Info, TensorState, Reward, Success, Termination, TimeOut, RobotState
from metasim.scenario.robot import RobotCfg
from metasim.utils.setup_util import get_sim_handler_class


class MasterSimulator:
    def __init__(
        self,
        scenario: ScenarioCfg | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        """Initialize the task env.

        Args:
            scenario: The scenario configuration. If None, it will use the class variable "scenario".
            device: The device to use for the environment. If None, it will use "cuda" if available, otherwise "cpu".
        """
        self.device : str = str(device) if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self._instantiate_env(scenario)
        self.initial_states: TensorState = self.handler.get_states()
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
        self.handler: BaseSimHandler = handler_class(scenario)
        self.handler.launch()

    def _physics_step(self, actions: Action) -> TensorState:
        """Physics step callback."""
        self.handler.set_dof_targets(actions)
        self.handler.simulate()
        env_states = self.handler.get_states()
        return env_states

    def close(self) -> None:
        """Close the environment."""
        self.handler.close()


class AgentEnv:
    """A base sub env for each embodiment in the env."""
    def __init__(self, simulator: MasterSimulator, robot: RobotCfg) -> None:
        """
        In this environment, we share the env_states for all robots, you can choose to do all/partial/no other robots' obs.
        """
        self._copy_shared_values(simulator, robot)
        self._episode_steps = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.max_episode_steps = -1 # no limit by default

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

        self.simulator.initial_states.robots[self.name]
        for key, value in vars(init_state).items():
            if value is not None:
                setattr(self.simulator.initial_states.robots[self.name], key, value)

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

    def reset(self, env_ids: list[int] = None) -> TensorState:
        """Reset the environment.

        Args:
            env_ids: The environment ids to reset
            states: Optional external states to set for the selected envs. If None, use initial states.

        Returns:
            TensorState: The TensorStateervation
            priv_TensorState: The privileged TensorStateervation
            info: The info
        """
        if env_ids is None:
            env_ids = list(range(self.num_envs))

        if len(env_ids) == 0:
            return self.simulator.handler.get_states()

        states_to_set = self.simulator.initial_states
        self.simulator.handler.set_states(states=states_to_set, env_ids=env_ids)
        env_states = self.simulator.handler.get_states(env_ids=env_ids)

        # reset episode step counters for reset envs
        ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)
        self._episode_steps[ids] = 0

        return env_states

    def step(self, actions: Action) -> tuple[TensorState, Reward, Success, TimeOut, Info | None]:
        raise NotImplementedError

    def _physics_step(self, actions) -> TensorState:
        return self.simulator._physics_step(actions)

    def _observation(self, env_states: TensorState) -> torch.Tensor:
        """Get the Observation of the environment."""
        raise NotImplementedError

    def _privileged_observation(self, env_states: TensorState) -> torch.Tensor:
        """Get the privileged Observation of the environment."""
        return None

    def _reward(self, env_states: TensorState) -> Reward:
        """Get the reward of the environment."""
        raise NotImplementedError

    def _terminated(self, env_states: TensorState) -> torch.BoolTensor:
        """Get the terminated of the environment."""
        raise NotImplementedError

    def _time_out(self, env_states: TensorState | None) -> torch.BoolTensor:
        """
        Timeout flags.
        Note that max_episode_steps is set to -1 by default (no timeout).
        """
        return self._episode_steps >= self.max_episode_steps

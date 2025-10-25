from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import asdict
from typing import Any

import torch

from metasim.scenario.scenario import ScenarioCfg
from metasim.task.rl_task import RLTaskEnv
from metasim.types import Action, Reward, TensorState
from roboverse_learn.rl.unitree_rl.configs import SensorsCfg


class AgentEnv(RLTaskEnv):
    """Base RLTaskEnv wrapper shared across Unitree locomotion embodiments."""

    def __init__(
        self,
        scenario: ScenarioCfg,
        config: Any,
        sensors: SensorsCfg | dict | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        self.cfg = config
        self._sensor_cfg = sensors
        self.extras: dict[str, Any] = {}

        # buffers will be allocated lazily once handler is available
        self.obs_buf_queue: deque[torch.Tensor] | None = None
        self.priv_obs_buf_queue: deque[torch.Tensor] | None = None
        self.actions: torch.Tensor | None = None
        self.torques: torch.Tensor | None = None
        self.rew_buf: torch.Tensor | None = None
        self.reset_buf: torch.Tensor | None = None
        self.time_out_buf: torch.Tensor | None = None

        self._initial_state_specs_cache: list[dict] | None = None

        super().__init__(scenario=scenario, device=device)

        self._initial_states_default = deepcopy(self._initial_states)
        self.name = self.robot.name if hasattr(self, "robot") else getattr(self, "name", None)

    # ------------------------------------------------------------------ #
    # RLTaskEnv hooks
    # ------------------------------------------------------------------ #
    def _build_initial_state_specs(self) -> list[dict]:
        """Return per-env dict used to seed simulator (override in subclasses)."""
        raise NotImplementedError

    def _get_initial_states(self) -> list[dict]:
        self._initial_state_specs_cache = self._build_initial_state_specs()
        return self._initial_state_specs_cache

    def _extra_spec(self) -> dict:
        """Expose optional sensor queries to the simulator handler."""
        if self._sensor_cfg is None:
            return {}
        if isinstance(self._sensor_cfg, dict):
            return self._sensor_cfg
        return asdict(self._sensor_cfg)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def get_states(self) -> TensorState:
        """Get the current simulator state."""
        return self.handler.get_states()

    def set_states(self, states: TensorState, env_ids: list[int] | None = None) -> None:
        """Set simulator state for selected env indices."""
        self.handler.set_states(states=states, env_ids=env_ids)

    def _physics_step(self, actions: Action) -> TensorState:
        """Issue low-level actions and simulate one physics step."""
        self.handler.set_dof_targets(actions)
        self.handler.simulate()
        return self.handler.get_states()

    def _reward(self, env_states: TensorState) -> Reward:
        raise NotImplementedError

    def _terminated(self, env_states: TensorState) -> torch.BoolTensor:
        raise NotImplementedError

    def _time_out(self, env_states: TensorState | None) -> torch.BoolTensor:
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Observation utilities
    # ------------------------------------------------------------------ #
    @property
    def obs_buf(self) -> torch.Tensor:
        """Stacked observation buffer with history along features."""
        if self.obs_buf_queue is None or len(self.obs_buf_queue) == 0:
            raise RuntimeError("Observation buffer not initialized.")
        return torch.cat(list(self.obs_buf_queue), dim=1)

    @property
    def priv_obs_buf(self) -> torch.Tensor:
        """Stacked privileged observation buffer with history along features."""
        if self.priv_obs_buf_queue is None or len(self.priv_obs_buf_queue) == 0:
            raise RuntimeError("Privileged observation buffer not initialized.")
        return torch.cat(list(self.priv_obs_buf_queue), dim=1)

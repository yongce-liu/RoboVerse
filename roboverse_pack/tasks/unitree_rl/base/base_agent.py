from __future__ import annotations

from collections import deque
from dataclasses import asdict
from typing import Any

import torch

from metasim.scenario.scenario import ScenarioCfg
from metasim.task.rl_task import RLTaskEnv
from metasim.types import Action, Reward, TensorState
from roboverse_learn.rl.unitree_rl.configs.cfg_base import BaseEnvCfg, CallbacksCfg


class AgentTask(RLTaskEnv):
    """Base RLTaskEnv wrapper shared across Unitree locomotion embodiments."""

    def __init__(
        self,
        scenario: ScenarioCfg,
        config: Any | BaseEnvCfg,
        device: str | torch.device | None = None,
    ) -> None:
        self.cfg = config
        _callbacks_cfg = asdict(getattr(self.cfg, "callbacks", CallbacksCfg()))
        self._query: dict = _callbacks_cfg.pop("query", {})
        super().__init__(scenario=scenario, device=device)

        # buffers will be allocated lazily once handler is available
        self.obs_buf_queue: deque[torch.Tensor] | None = None
        self.priv_obs_buf_queue: deque[torch.Tensor] | None = None
        self.actions: torch.Tensor | None = None
        # self.torques: torch.Tensor | None = None
        self.rew_buf: torch.Tensor | None = None
        self.reset_buf: torch.Tensor | None = None
        # self.time_out_buf: torch.Tensor | None = None
        self.extras: dict[str, Any] = {}

        # Callbacks
        self._bind_callbacks(callbacks=_callbacks_cfg)

    def _bind_callbacks(self, callbacks: dict | None = None):
        for _callback_key, _callbacks in callbacks.items():
            for _key, _val in _callbacks.items():
                if not isinstance(_val, tuple):
                    _callbacks[_key] = (_val, {})
                if hasattr(_callbacks[_key][0], "bind_handler"):
                    _callbacks[_key][0].bind_handler(self.handler)

        _setup_callbacks = callbacks.pop("setup", {})
        for _key, _val in _setup_callbacks.items():
            _val[0](**_val[1])  ## call itself
        self._reset_callbacks = callbacks.pop("reset", {})
        self._step_callbacks = callbacks.pop("step", {})
        self._terminate_callbacks = callbacks.pop("terminate", {})
        self.episode_not_terminations = {}
        for _key in self._terminate_callbacks.keys():
            self.episode_not_terminations[_key] = torch.zeros(
                size=(self.num_envs,), dtype=torch.float, device=self.device
            )

    # ------------------------------------------------------------------ #
    # RLTaskEnv hooks
    # ------------------------------------------------------------------ #
    def _get_initial_states(self):
        raise NotImplementedError

    def _extra_spec(self) -> dict:
        """Expose optional sensor queries to the simulator handler."""
        return self._query

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
        self.handler.simulate(decimation=1)  # decimation control in task_env level
        return self.handler.get_states()

    def _reward(self, env_states: TensorState) -> Reward:
        raise NotImplementedError

    def _terminated(self, env_states: TensorState) -> torch.BoolTensor:
        raise NotImplementedError

    def _time_out(self, env_states: TensorState | None) -> torch.BoolTensor:
        raise NotImplementedError

    def _observation(self, env_states):
        # return super()._observation(env_states) --- IGNORE ---
        pass

    def _privileged_observation(self, env_states):
        # return super()._privileged_observation(env_states) --- IGNORE ---
        pass

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

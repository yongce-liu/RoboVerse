from __future__ import annotations

from typing import Dict

import torch

from metasim.queries.base import BaseQueryType


class NetContactForce(BaseQueryType):
    """Optional query to fetch per-body net contact forces for each robot.

    - For IsaacGym: uses the native net-contact tensor and maps it per-robot in handler indexing order.
    - For IsaacSim: returns a zero tensor fallback per-robot (hook is in place; replace with real source when available).
    """

    # Supported handlers by module path
    supported_handlers = [
        "metasim.sim.isaacgym.isaacgym",
        "metasim.sim.isaacsim.isaacsim",
    ]

    def __init__(self):
        super().__init__()

    def bind_handler(self, handler, *args, **kwargs):
        super().bind_handler(handler, *args, **kwargs)

    def _for_isaacgym(self) -> Dict[str, torch.Tensor]:
        # Ensure source tensor exists
        from isaacgym import gymtorch

        contact_forces = gymtorch.wrap_tensor(
            self.handler.gym.acquire_net_contact_force_tensor(self.handler.sim)
        )
        # Refresh once to populate
        # self.handler.gym.refresh_net_contact_force_tensor(self.handler.sim)
        return contact_forces

    def _for_isaacsim(self) -> Dict[str, torch.Tensor]:
        # only for single robot envs for now
        robot_name = self.handler.contact_sensor.cfg.prim_path.split("/")[-2]
        contact_forces = self.handler.contact_sensor.data.net_forces_w
        return {robot_name: contact_forces}

    def __call__(self):
        mod = self.handler.__class__.__module__
        if mod.startswith("metasim.sim.isaacgym"):
            return self._for_isaacgym()
        elif mod.startswith("metasim.sim.isaacsim"):
            return self._for_isaacsim()
        else:  # pragma: no cover - other handlers not yet supported
            raise ValueError(f"Unsupported handler type: {type(self.handler)} for NetContactForce query")

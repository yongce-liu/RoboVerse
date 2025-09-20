from __future__ import annotations
from typing import Dict

import torch

from metasim.sim.base import BaseSimHandler, BaseQueryType
from metasim.utils import configclass
import numpy as np

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

try:
    import mujoco  # noqa: F401
except ImportError:
    pass
class ContactForces(BaseQueryType):
    """Optional query to fetch per-body net contact forces for each robot.

    - For IsaacGym: uses the native net-contact tensor and maps it per-robot in handler indexing order.
    - For IsaacSim: returns a zero tensor fallback per-robot (hook is in place; replace with real source when available).
    """
    def __init__(self):
        super().__init__()

    def bind_handler(self, handler:BaseSimHandler, *args, **kwargs):
        super().bind_handler(handler, *args, **kwargs)
        self.simulator = handler.scenario.simulator
        self.num_envs = handler.scenario.num_envs
        self.robots = handler.robots
        self.body_ids_reindex = handler._get_body_ids_reindex(self.robots[0].name) if hasattr(self.handler, '_get_body_ids_reindex') else handler.get_body_reindex(self.robots[0].name)
        self.initialize()

    def initialize(self):
        if self.simulator == "isaacgym":
            self.contact_forces = isaacgym.gymtorch.wrap_tensor(self.handler.gym.acquire_net_contact_force_tensor(self.handler.sim))
        elif self.simulator == "isaacsim":
            self.contact_forces = self.handler.contact_sensor.data.net_forces_w
        elif self.simulator == "mujoco":
            self.contact_forces = self._get_contact_forces_mujoco()
        else:
            raise NotImplementedError

    def _get_contact_forces_mujoco(self) -> torch.Tensor:
        """
        Compute net contact forces on each body.
        Returns:
            torch.Tensor: shape (nbody, 3), contact forces for each body
        """
        import mujoco
        nbody = self.handler.physics.model.nbody
        contact_forces = torch.zeros((nbody, 3), device=self.handler.device)

        for i in range(self.handler.physics.data.ncon):
            contact = self.handler.physics.data.contact[i]
            force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.handler.physics.model.ptr, self.handler.physics.data.ptr, i, force)
            f_contact = torch.from_numpy(force[:3]).to(device=self.handler.device)

            body1 = self.handler.physics.model.geom_bodyid[contact.geom1]
            body2 = self.handler.physics.model.geom_bodyid[contact.geom2]

            contact_forces[body1] += f_contact
            contact_forces[body2] -= f_contact

        return contact_forces


    def __call__(self):
        if self.simulator == "isaacgym":
            self.handler.gym.refresh_net_contact_force_tensor(self.handler.sim)
        elif self.simulator == "isaacsim":
            self.contact_forces = self.handler.contact_sensor.data.net_forces_w
        elif self.simulator == "mujoco":
            self.contact_forces = self._get_contact_forces_mujoco()
        else:
            raise NotImplementedError
        return {self.robots[0].name: self.contact_forces.view(self.num_envs, -1, 3)[:, self.body_ids_reindex, :]}

@configclass
class SensorsCfg:
    contact_forces: ContactForces = ContactForces()

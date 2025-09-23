from metasim.constants import PhysicStateType
from metasim.scenario.objects import PrimitiveSphereCfg


class BallCfg(PrimitiveSphereCfg):
    def __init__(self):
        super().__init__(
            name="ball",
            radius=0.1,
            color=[1.0, 0.0, 1.0],
            physics=PhysicStateType.RIGIDBODY,
        )
        self.mass = 0.1
        self.enabled_gravity = True
        self.collision_enabled = False
        self.init_position = [0.6, 0.0, 1.0]
        self.init_rotation = [1.0, 0.0, 0.0, 0.0]  # w, x, y, z

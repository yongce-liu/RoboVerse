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

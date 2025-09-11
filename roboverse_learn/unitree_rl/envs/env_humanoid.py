from .env_legged_robot import LeggedRobotEnv

class HumanoidEnv(LeggedRobotEnv):
    def __init__(self, simulator, robot):
        super().__init__(simulator, robot)
        # additional humanoid specific initialization can go here

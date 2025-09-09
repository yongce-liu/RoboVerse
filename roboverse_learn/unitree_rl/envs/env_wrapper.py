from metasim.task.base import BaseTaskEnv

class RslRLWrapper:
    def __init__(self, env: BaseTaskEnv):
        self.env = env


class SB3Wrapper:
    def __init__(self, env: BaseTaskEnv):
        self.env = env


class RlLibWrapper:
    def __init__(self, env: BaseTaskEnv):
        self.env = env

def make_env():
    pass

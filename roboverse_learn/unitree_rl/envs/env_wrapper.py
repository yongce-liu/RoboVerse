from glob import glob
import random
import shutil
import sys
import pickle as pkl
import torch
import os
import numpy as np
from metasim.task.base import BaseTaskEnv
from roboverse_learn.unitree_rl.helper.utils import get_class, get_log_dir
from roboverse_learn.unitree_rl.envs.env_base import MasterSimulator
from roboverse_learn.unitree_rl.configs.cfg_base import BaseEnvCfg
from roboverse_learn.unitree_rl.envs.env_base import AgentEnv


def set_seed(seed):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print(f"Setting seed: {seed}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



class RslRLEnvWrapper:
    def __init__(self, env: AgentEnv):
        self.env = env

    @property
    def num_privileged_obs(self):
        return self.env.num_priv_obs

class SB3EnvWrapper:
    def __init__(self, env: BaseTaskEnv):
        self.env = env

class RlLibEnvWrapper:
    def __init__(self, env: BaseTaskEnv):
        self.env = env

class BaseWrapper:
    def __init__(self, env: BaseTaskEnv, train_cfg, log_dir:str):
        raise NotImplementedError

    def load(self, path):
        raise NotImplementedError

    def learn(self):
        raise NotImplementedError

class RslRlWrapper(BaseWrapper):
    def __init__(self, env: BaseTaskEnv, train_cfg, log_dir:str):
        from rsl_rl.runners.on_policy_runner import OnPolicyRunner
        self.env = RslRLEnvWrapper(env)
        self.device = env.device
        if not isinstance(train_cfg, dict):
            train_cfg = train_cfg.__dict__
        self.train_cfg = train_cfg
        self.log_dir = log_dir

        self.runner = OnPolicyRunner(
            env=self.env,
            train_cfg=self.train_cfg,
            device=self.device,
            log_dir=log_dir,
        )

    def learn(self, max_iterations=None):
        self.runner.learn(num_learning_iterations=max_iterations, init_at_random_ep_len=True)

    def load(self, path):
        self.runner.load(path)

class Runner:
    def __init__(self,
                 simulator: MasterSimulator,
                 task_name: str,
                 lib_name: str = "rsl_rl"):
        set_seed(0)
        self.task_name = task_name
        self.runners = {}
        base_lib_path = "roboverse_learn.unitree_rl.tasks"
        env_cls = get_class(task_name, suffix="Env", library=base_lib_path)
        env_cls_path = sys.modules[env_cls.__module__].__file__
        env_cfg_cls: BaseEnvCfg = get_class(task_name, suffix="EnvCfg", library=base_lib_path)
        train_cfg_cls = get_class(task_name+"_"+lib_name, suffix="TrainCfg", library=base_lib_path)
        runner_cls = get_class(lib_name, suffix="Wrapper", library="roboverse_learn.unitree_rl.envs.env_wrapper")
        # construct the separate environment for each embodiment
        for _robot in simulator.robots:
            log_dir = get_log_dir(task_name=task_name, robot_name=_robot.name)
            # backup
            shutil.copy2(env_cls_path, log_dir)
            env_cfg = env_cfg_cls()
            pkl.dump(env_cfg, open(f"{log_dir}/env_cfg.pkl", "wb"))
            env: AgentEnv = env_cls(simulator=simulator, robot=_robot, config=env_cfg)
            train_cfg = train_cfg_cls()
            pkl.dump(train_cfg, open(f"{log_dir}/train_cfg.pkl", "wb"))
            runner: BaseWrapper = runner_cls(env=env, train_cfg=train_cfg, log_dir=log_dir)
            self.runners[_robot.name] = runner

    def learn(self):
        tmp_runner = self.runners[list(self.runners.keys())[0]]
        tmp_runner.learn()

    def load(self, resume_dir: str, checkpoint: int = None):
        for _robot_name, _runner in self.runners.items():
            log_dir = get_log_dir(task_name=self.task_name, robot_name=_robot_name, now=resume_dir)
            if checkpoint is None:
                all_checkpoints = glob(os.path.join(log_dir, "model_*.pt"))
                if len(all_checkpoints) == 0:
                    raise FileNotFoundError(f"No checkpoints found in {log_dir}")
                all_checkpoints = [int(p.split("model_")[-1].split(".pt")[0]) for p in all_checkpoints]
                checkpoint = max(all_checkpoints)
                print(f"Loading the latest checkpoint: {checkpoint}")
            _runner.load(os.path.join(log_dir, f"model_{checkpoint}.pt"))

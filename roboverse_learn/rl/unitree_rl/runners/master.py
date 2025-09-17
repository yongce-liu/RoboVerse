from __future__ import annotations
from typing import Union

import sys
import shutil
import datetime
import pickle as pkl

from roboverse_learn.rl.unitree_rl.configs.cfg_base import BaseEnvCfg
from roboverse_learn.rl.unitree_rl.envs import MasterSimulator, EnvTypes
from roboverse_learn.rl.unitree_rl.helper import get_class, get_log_dir, get_load_path


class BaseRunnerWrapper:
    def __init__(self, env: EnvTypes, train_cfg: dict, log_dir: str):
        self.env = env
        self.device = env.device
        if not isinstance(train_cfg, dict):
            train_cfg = train_cfg.to_dict()
        self.train_cfg = train_cfg
        self.log_dir = log_dir

    def load(self, path):
        raise NotImplementedError

    def learn(self, max_iterations):
        raise NotImplementedError

    def get_policy(self):
        raise NotImplementedError



class MasterRunner:
    def __init__(self,
                 simulator: MasterSimulator,
                 task_name: str,
                 log_path: str = None,
                 lib_name: str = "rsl_rl"):
        self.task_name = task_name
        self.runners = {}
        base_lib_path = "roboverse_learn.unitree_rl.tasks"
        env_cls = get_class(task_name, suffix="Env", library=base_lib_path)
        env_cfg_cls: BaseEnvCfg = get_class(task_name, suffix="EnvCfg", library=base_lib_path)
        train_cfg_cls = get_class(task_name+"_"+lib_name, suffix="TrainCfg", library=base_lib_path)
        runner_cls = get_class(lib_name, suffix="Wrapper", library="roboverse_learn.unitree_rl.runners")
        # construct the separate environment for each embodiment
        # FOR BACKUP
        env_cls_path = sys.modules[env_cls.__module__].__file__
        now = log_path if log_path else datetime.datetime.now().strftime("%Y_%m%d_%H%M%S")
        for _robot in simulator.robots:
            log_dir = get_log_dir(task_name=task_name, robot_name=_robot.name, now=now)
            env_cfg = env_cfg_cls()
            env: EnvTypes = env_cls(simulator=simulator, robot=_robot, config=env_cfg)
            train_cfg = train_cfg_cls()
            runner: BaseRunnerWrapper = runner_cls(env=env, train_cfg=train_cfg, log_dir=log_dir)
            self.runners[_robot.name] = runner
            shutil.copy2(env_cls_path, log_dir)
            pkl.dump(env_cfg, open(f"{log_dir}/env_cfg.pkl", "wb"))
            pkl.dump(train_cfg, open(f"{log_dir}/train_cfg.pkl", "wb"))

    def learn(self, max_iterations=10000):
        tmp_runner = self.runners[list(self.runners.keys())[0]]
        tmp_runner.learn(max_iterations=max_iterations)

    def load(self, resume_dir: str, checkpoint: int = None):
        self.policys = {}
        for _robot_name, _runner in self.runners.items():
            log_dir = get_log_dir(task_name=self.task_name, robot_name=_robot_name, now=resume_dir)
            _runner.load(get_load_path(load_root=log_dir, checkpoint=checkpoint))
            self.policys[_robot_name] = _runner.get_policy()
        return self.policys

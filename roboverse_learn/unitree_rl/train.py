from __future__ import annotations

import os
try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import rootutils
rootutils.setup_root(__file__, pythonpath=True)
import torch
import wandb
import random
import shutil
import numpy as np
from loguru import logger as log

from metasim.scenario.scenario import ScenarioCfg
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

from roboverse_learn.unitree_rl.envs.env_base import MasterSimulator, AgentEnv
from roboverse_learn.unitree_rl.configs.cfg_base import BaseEnvCfg
from roboverse_learn.unitree_rl.envs.env_wrapper import make_runner
from roboverse_learn.unitree_rl.helper.utils import get_args, get_class, get_log_dir, make_robots


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


def train(args):
    # only support single robot for now
    _robots_name, _robots = make_robots(args)
    robots_name, robots = [_robots_name[0]], [_robots[0]]
    task_config: BaseEnvCfg = get_class(args.task, suffix="Cfg")()

    scenario = ScenarioCfg(
        robots=robots,
        num_envs=args.num_envs,
        simulator=args.sim,
        renderer=args.sim,
        headless=args.headless,
        sim_params=task_config.sim_params,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    master_simulator = MasterSimulator(scenario=scenario, device=args.device)
    robot0_env = get_class(args.task, suffix="Env")(simulator=master_simulator, robot=master_simulator.robots[0], config=task_config)
    train_cfg = get_class(args.task, suffix="TrainCfg")()
    runner = make_runner(env=robot0_env, train_cfg=train_cfg, lib='rsl')

    runner.learn()

if __name__ == "__main__":
    # set_seed(1)
    args = get_args()
    # args.task = "dof12_walking"
    # args.sim = "isaacgym"
    # args.num_envs = 1
    # args.robot = "g1_dof12"
    # args.headless = True
    train(args)



try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import rootutils
rootutils.setup_root(__file__, pythonpath=True)

import os
import torch
import wandb
import random
import numpy as np

from rsl_rl.runners.on_policy_runner import OnPolicyRunner
from metasim.scenario.scenario import ScenarioCfg


from metasim.scenario.scenario import ScenarioCfg
from roboverse_learn.unitree_rl.envs.env_base import MasterSimulator
from roboverse_learn.unitree_rl.configs.cfg_base import RslTrainCfg
from roboverse_learn.unitree_rl.tasks.walking_dof12 import WalkingDof12Env, WalkingDof12Cfg
from roboverse_learn.unitree_rl.helper.utils import (
    PolicyExporterLSTM,
    export_policy_as_jit,
    get_args,
    get_class,
    get_export_jit_path,
    get_load_path,
    make_robots,
    reindex_func,
)



def train(args):
    # only support single robot for now
    _robots_name, _robots = make_robots(args)
    robots_name, robots = [_robots_name[0]], [_robots[0]]
    device = 'cpu' if args.sim =="mujoco" else "cuda" if torch.cuda.is_available() else "cpu"
    task_config = get_class(args.task, suffix="Cfg")()
    scenario = ScenarioCfg(
        robots=robots,
        num_envs=args.num_envs,
        simulator=args.sim,
        renderer=args.sim,
        headless=args.headless,
        sim_params=task_config.sim_params,
        )

    # Use the existing run directory as log_dir to avoid creating new output dirs during play
    log_dir = "/home/air/RoboVerse/outputs/unitree_rl/g1_dof12_dof12_walking/tmp/"
    load_path = "/home/air/RoboVerse/outputs/unitree_rl/g1_dof12_dof12_walking/final/exported/model_exported_jit.pt"

    master = MasterSimulator(scenario=scenario, device=device)
    config = WalkingDof12Cfg()
    config.commands.curriculum = False
    config.commands.resampling_time = 1e6  # effectively disable command changes
    config.domain_rand.randomize_friction = False
    config.domain_rand.randomize_base_mass = False
    config.domain_rand.randomize_initial_state = False
    config.domain_rand.push_robots = False
    config.noise.add_noise = False
    train_config = RslTrainCfg()
    env = WalkingDof12Env(simulator=master, robot=robots[0], config=config)

    _ = env.reset()
    obs = env.step(torch.zeros(size=(env.num_envs, env.num_actions), device=env.device))[0]

    # # load policy
    # ppo_runner = OnPolicyRunner(
    #     env=env,
    #     train_cfg=train_config,
    #     device=env.device,
    #     log_dir=log_dir,
    # )
    policy = torch.jit.load(load_path).to(env.device)


    for i in range(1000000):
        # set fixed command
        env.commands[:, 0] = 0.5
        env.commands[:, 1] = 0.0
        env.commands[:, 2] = 0.0
        env.commands[:, 3] = 0.0
        actions = policy(obs.detach()).detach()
        obs, _, _, _, _ = env.step(actions)

if __name__ == "__main__":
    # set_seed(1)
    args = get_args()
    args.task = "walking_dof12"
    args.sim = "mujoco"
    args.num_envs = 1
    args.robot = "g1_dof12"
    # args.headless = True
    train(args)

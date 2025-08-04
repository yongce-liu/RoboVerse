from __future__ import annotations

import os
import shutil

from loguru import logger as log

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import rootutils
import torch

rootutils.setup_root(__file__, pythonpath=True)

import wandb
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

from metasim.cfg.scenario import ScenarioCfg
from metasim.utils.setup_util import get_robot
from roboverse_learn.unitree_rl.utils import get_args, get_log_dir, get_class

def make_robots(args):
    robots_name = args.robot.replace(" ", "").replace("[", "").replace("]", "").replace("'", "").replace('"', '').split(",")
    print(robots_name)
    robots = []
    for _name in robots_name:
        print(_name)
        robot = get_robot(_name)
        robots.append(robot)
    return robots


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # only support single robot for now
    robots = [make_robots(args)[0]]
    config_wrapper = get_class(args.task, "Cfg")
    task = config_wrapper(robots=robots)
    scenario = ScenarioCfg(
        task=task,
        robots=robots,
        num_envs=args.num_envs,
        sim=args.sim,
        headless=args.headless,
        cameras=[]
    )

    use_wandb = args.use_wandb
    if use_wandb:
        wandb.init(project=args.wandb, name=args.run_name)

    if args.load_run:
        datetime = args.load_run.split("/")[-2]
    else:
        datetime = None
    log_dir = get_log_dir(args, scenario, datetime)
    task_wrapper = get_class(args.task, "Task")
    env = task_wrapper(scenario)

    # dump snapshot of training config
    task_path = f"roboverse_learn/unitree_rl/configs/{scenario.task.task_name}.py"
    if not os.path.exists(task_path):
        log.error(f"Task path {task_path} does not exist, please check your task name in config carefully")
        return
    shutil.copy2(task_path, log_dir)

    ppo_runner = OnPolicyRunner(
        env=env,
        train_cfg=env.train_cfg,
        device=device,
        log_dir=log_dir,
        # wandb=use_wandb,
        args=args,
    )
    if args.load_run:
        ppo_runner.load(args.load_run)
    ppo_runner.learn(num_learning_iterations=args.learning_iterations)


if __name__ == "__main__":
    args = get_args()
    train(args)

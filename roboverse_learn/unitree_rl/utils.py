from __future__ import annotations

import argparse
import copy
import datetime
import importlib
import os

import torch
from loguru import logger as log

from metasim.cfg.scenario import ScenarioCfg
from metasim.sim import BaseSimHandler
from metasim.utils import is_camel_case, is_snake_case, to_camel_case


def parse_arguments(description="humanoid rl task arguments", custom_parameters=None):
    """Parse command line arguments."""

    if custom_parameters is None:
        custom_parameters = []
    parser = argparse.ArgumentParser(description=description)
    for argument in custom_parameters:
        if ("name" in argument) and ("type" in argument or "action" in argument):
            help_str = ""
            if "help" in argument:
                help_str = argument["help"]

            if "type" in argument:
                if "default" in argument:
                    parser.add_argument(
                        argument["name"], type=argument["type"], default=argument["default"], help=help_str
                    )
                else:
                    parser.add_argument(argument["name"], type=argument["type"], help=help_str)
            elif "action" in argument:
                parser.add_argument(argument["name"], action=argument["action"], help=help_str)

        else:
            log.error("ERROR: command line argument name, type/action must be defined, argument not added to parser")
            log.error("supported keys: name, type, default, action, help")

    return parser.parse_args()

def get_args(test=False):
    """Get the command line arguments."""

    custom_parameters = [
        {
            "name": "--task",
            "type": str,
            "default": "HumanoidWalking",
            "help": "Resume training or start testing from a checkpoint. Overrides config file if provided.",
        },
        {
            "name": "--robot",
            "type": str,
            "default": "h1",
            "help": "Resume training or start testing from a checkpoint. Overrides config file if provided.",
        },
        {
            "name": "--num_envs",
            "type": int,
            "default": 128,
            "help": "number of parallel environments.",
        },
        {
            "name": "--sim",
            "type": str,
            "default": "isaacgym",
            "help": "simulator type, currently only isaacgym is supported",
        },
        {"name": "--resume", "action": "store_true", "default": False, "help": "Resume training from a checkpoint"},
        {
            "name": "--run_name",
            "type": str,
            "required": True if not test else False,
            "help": "Name of the run. Overrides config file if provided.",
        },
        {
            "name": "--learning_iterations",
            "type": int,
            "default": 15000,
            "help": "Path to the config file. If provided, will override command line arguments.",
        },
        {
            "name": "--load_run",
            "type": str,
            "default": None,
            "help": "Path to the config file. If provided, will override command line arguments.",
        },
        {
            "name": "--checkpoint",
            "type": int,
            "default": -1,
            "help": "Saved model checkpoint number. If -1: will load the last checkpoint. Overrides config file if provided.",
        },
        {"name": "--headless", "action": "store_true", "default": True, "help": "Force display off at all times"},
        {"name": "--use_wandb", "action": "store_true", "default": True, "help": "Use wandb for logging"},
        {"name": "--wandb", "type": str, "default": "g1_walking", "help": "Wandb project name"},
    ]
    args = parse_arguments(custom_parameters=custom_parameters)
    return args

def get_log_dir(args: argparse.Namespace, scenario: ScenarioCfg, now: str) -> str:
    """Get the log directory."""

    robot_name = args.robot
    task_name = scenario.task.task_name
    task_name = f"{robot_name}_{task_name}"
    if now is not None:
        now = datetime.datetime.now().strftime("%Y_%m%d_%H%M%S")
    log_dir = f"./outputs/unitree_rl/{task_name}/{now}/"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    log.info("Log directory: {}", log_dir)
    return log_dir

def get_class(name: str, suffix=""):
    """Get the environment wrapper class for the given task ID."""
    if is_camel_case(name):
        task_name_camel = name

    elif is_snake_case(name):
        task_name_camel = to_camel_case(name)

    wrapper_module = importlib.import_module("roboverse_learn.unitree_rl")
    wrapper_cls = getattr(wrapper_module, f"{task_name_camel}{suffix}")
    return wrapper_cls

def get_body_reindexed_indices_from_substring(
    sim_handler: BaseSimHandler, obj_name: str, body_names: list[str], device
):
    """given substrings of body name, find all the bodies indices in sorted order."""

    matches = []
    sorted_names = sim_handler.get_body_names(obj_name, sort=True)

    for name in body_names:
        for i, s in enumerate(sorted_names):
            if name in s:
                matches.append(i)

    index = torch.tensor(matches, dtype=torch.int32, device=device)
    return index

def get_joint_reindexed_indices_from_substring(
    sim_handler: BaseSimHandler, obj_name: str, joint_names: list[str], device: str
):
    """given substrings of joint name, find all the bodies indices in sorted order."""

    matches = []
    sorted_names = sim_handler.get_joint_names(obj_name, sort=True)

    for name in joint_names:
        for i, s in enumerate(sorted_names):
            if name in s:
                matches.append(i)

    index = torch.tensor(matches, dtype=torch.int32, device=device)
    return index

def torch_rand_float(lower: float, upper: float, shape: tuple[int, int], device: str) -> torch.Tensor:
    """Generate a tensor of random floats in the range [lower, upper]."""
    return (upper - lower) * torch.rand(*shape, device=device) + lower

def wrap_to_pi(angles: torch.Tensor) -> torch.Tensor:
    angles %= 2 * torch.pi
    angles -= 2 * torch.pi * (angles > torch.pi)
    return angles

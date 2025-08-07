import os
import rootutils
PROJECT_ROOT_DIR = str(rootutils.setup_root(__file__, pythonpath=True))
UNITREE_GYM_ROOT_DIR = os.path.join(PROJECT_ROOT_DIR, 'roboverse_learn', 'unitree_rl')

from roboverse_learn.unitree_rl.deploy.mujoco_utils import IndependentMujocoController
from metasim.cfg.scenario import ScenarioCfg
from roboverse_learn.unitree_rl.utils import (
    get_class,
    make_robots,
)

from roboverse_learn.unitree_rl.utils import parse_arguments
import numpy as np



def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


if __name__ == "__main__":
    # get config file name from command line
    import argparse

    parser = argparse.ArgumentParser()
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
            "default": "g1",
            "help": "Resume training or start testing from a checkpoint. Overrides config file if provided.",
        },
        {"name": "--headless", "action": "store_true", "default": True, "help": "Force display off at all times"},
        {"name": "--config_file", "type": str, "default": "g1.yaml", "help": "config file name in the config folder"}
    ]

    args = parse_arguments(custom_parameters=custom_parameters)
    robot_names, robots = [make_robots(args)[0]]
    config_wrapper = get_class(args.task, "Cfg")
    task = config_wrapper(robots=robots)
    scenario = ScenarioCfg(
        task=task,
        robots=[args.robot],
        num_envs=1,
        sim="mujoco",
        headless=args.headless,
        cameras=[],
    )
    scenario.num_envs = 1
    scenario.task.commands.curriculum = False
    scenario.task.ppo_cfg.runner.resume = True
    scenario.task.random.friction.enabled = False
    scenario.task.random.mass.enabled = False
    scenario.task.random.push.enabled = False

    controller = IndependentMujocoController(args, scenario)
    controller.launch()

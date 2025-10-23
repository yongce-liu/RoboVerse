from __future__ import annotations

import copy

import rootutils

rootutils.setup_root(__file__, pythonpath=True)

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import torch

from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import get_task_class

from roboverse_pack.tasks.unitree_rl.envs import EnvTypes
from roboverse_learn.rl.unitree_rl.helper import get_args, make_objects, make_robots, set_seed
from roboverse_learn.rl.unitree_rl.runners import EnvWrapperTypes, MasterRunner


def prepare(args):
    task_cls = get_task_class(args.task)
    scenario_template = getattr(task_cls, "scenario", ScenarioCfg())
    scenario = copy.deepcopy(scenario_template)

    overrides = {
        "num_envs": args.num_envs,
        "simulator": args.sim,
        "headless": args.headless,
    }

    if args.robots:
        overrides["robots"] = make_robots(args.robots)
        overrides["cameras"] = [
            camera
            for robot in overrides["robots"]
            if hasattr(robot, "cameras")
            for camera in getattr(robot, "cameras", [])
        ]

    if args.objects:
        overrides["objects"] = make_objects(args.objects)

    scenario.update(**overrides)

    device = "cpu" if args.sim == "mujoco" else ("cuda" if torch.cuda.is_available() else "cpu")

    master_runner = MasterRunner(
        task_cls=task_cls,
        scenario=scenario,
        log_path=args.resume,
        lib_name="rsl_rl",
        device=device,
    )

    return master_runner


def play(args):
    master_runner = prepare(args)
    if args.resume:
        policys = master_runner.load(resume_dir=args.resume, checkpoint=args.checkpoint)
    else:
        raise ValueError("Please provide the resume dir for eval policy.")

    name_0 = list(master_runner.runners.keys())[0]
    runner_0 = master_runner.runners[name_0]
    policy_0 = policys[name_0]
    env_0: EnvTypes = runner_0.env
    envwrapper_0: EnvWrapperTypes = runner_0.env_wrapper
    cfg_0 = env_0.cfg

    cfg_0.commands.curriculum = False
    cfg_0.commands.resampling_time = 1e6  # effectively disable command changes
    cfg_0.domain_rand.randomize_friction = False
    cfg_0.domain_rand.randomize_base_mass = False
    cfg_0.domain_rand.randomize_initial_state = False
    cfg_0.domain_rand.push_robots = False
    cfg_0.noise.add_noise = False

    # unenable noise and randomization for eval

    env_0.reset()
    obs, _, _, _, _ = env_0.step(torch.zeros(env_0.num_envs, env_0.num_actions, device=env_0.device))
    obs = envwrapper_0.get_observations()


    for i in range(1000000):
        # set fixed command
        env_0.commands[:, 0] = 0.5
        env_0.commands[:, 1] = 0.0
        env_0.commands[:, 2] = 0.0
        env_0.commands[:, 3] = 0.0
        actions = policy_0(obs)
        obs, _, _, _ = envwrapper_0.step(actions)

def train(args):
    master_runner = prepare(args)
    if args.resume:
        master_runner.load(resume_dir=args.resume, checkpoint=args.checkpoint)
    master_runner.learn(max_iterations=args.iter)

if __name__ == "__main__":
    args = get_args()
    # args.task = "walking_dof12"
    # args.sim = "mujoco"
    # args.num_envs = 1
    # args.robot = "g1_dof12"
    # args.headless = True
    # args.seed = 0
    # args.resume = "2025_0920_075050"
    # args.eval = True
    set_seed(args.seed)
    if args.eval:
        play(args)
    else:
        train(args)

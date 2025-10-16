from __future__ import annotations

import rootutils
rootutils.setup_root(__file__, pythonpath=True)

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import torch

from metasim.scenario.scenario import ScenarioCfg
from metasim.scenario.lights import DomeLightCfg, DistantLightCfg, DiskLightCfg
from metasim.scenario.simulator_params import SimParamCfg

from roboverse_learn.rl.unitree_rl.configs import SensorsCfg
from roboverse_pack.tasks.unitree_rl.envs import MasterSimulator, EnvTypes
from roboverse_learn.rl.unitree_rl.runners import MasterRunner, EnvWrapperTypes
from roboverse_learn.rl.unitree_rl.helper import get_args, make_robots, set_seed, make_objects


def prepare(args):
    # only support single robot for now
    robots: list = [make_robots(args.robots)[0]]  # get the first robot
    objects: list = make_objects(args.objects) if args.objects is not None else []

    # should move the parameters in a common used config files
    env_spacing = 2.5
    # decimation = 4
    device = "cpu" if args.sim == "mujoco" else "cuda" if torch.cuda.is_available() else "cpu"
    sim_params = SimParamCfg(dt=0.005,
                            substeps=1,
                            num_threads=10,
                            solver_type=1,
                            num_position_iterations=4,
                            num_velocity_iterations=0,
                            contact_offset=0.01,
                            rest_offset=0.0,
                            bounce_threshold_velocity=0.5,
                            max_depenetration_velocity=1.0,
                            default_buffer_size_multiplier=5,
                            replace_cylinder_with_capsule=True,
                            friction_correlation_distance=0.025,
                            friction_offset_threshold=0.04)
    # should move the parameters in a common used config files

    scenario = ScenarioCfg(
        robots=robots,
        objects=objects,
        cameras=[camera for robot in robots if hasattr(robot, 'cameras')
         for camera in robot.cameras],
        num_envs=args.num_envs,
        simulator=args.sim,
        # renderer=args.sim,
        headless=args.headless,
        env_spacing=env_spacing,
        sim_params=sim_params,
        # decimation=decimation,
        lights=[
                # Sky dome light - provides soft ambient lighting from all directions
                DomeLightCfg(
                    intensity=800.0,  # Moderate ambient lighting
                    color=(0.85, 0.9, 1.0),  # Slightly blue sky color
                )
        ]
    )

    sensors = SensorsCfg()
    master_simulator = MasterSimulator(scenario=scenario, sensors=sensors, device=device)
    master_runner = MasterRunner(simulator=master_simulator, log_path=args.resume ,task_name=args.task, lib_name='rsl_rl')

    return master_runner


def play(args):
    master_runner = prepare(args)
    if args.resume:
        policys = master_runner.load(resume_dir=args.resume, checkpoint=-1)
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

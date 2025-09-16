from __future__ import annotations

import rootutils
rootutils.setup_root(__file__, pythonpath=True)

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import torch

from metasim.scenario.scenario import ScenarioCfg
from metasim.scenario.simulator_params import SimParamCfg

from roboverse_learn.unitree_rl.envs import MasterSimulator, Runner
from roboverse_learn.unitree_rl.helper import get_args, make_robots, set_seed


def train(args):
    # only support single robot for now
    robots: list = [make_robots(args.robots)[0]]  # get the first robot

    # should move the parameters in a common used config files
    env_spacing = 4
    # decimation = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
        num_envs=args.num_envs,
        simulator=args.sim,
        renderer=args.sim,
        headless=args.headless,
        env_spacing=env_spacing,
        sim_params=sim_params,
        # decimation=decimation,
    )

    master_simulator = MasterSimulator(scenario=scenario, device=device)
    runner = Runner(simulator=master_simulator, task_name=args.task, lib_name='rsl_rl')
    if args.resume:
        runner.load(resume_dir=args.resume, checkpoint=args.checkpoint)
    runner.learn(max_iterations=args.iter)

if __name__ == "__main__":
    args = get_args()
    # args.task = "walking_dof12"
    # args.sim = "isaacgym"
    # args.num_envs = 1
    # args.robot = "g1_dof12"
    # args.headless = True
    # args.seed = 0
    set_seed(args.seed)
    train(args)

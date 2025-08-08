import rootutils

rootutils.setup_root(__file__, pythonpath=True)
try:
    import isaacgym  # noqa: F401
except ImportError:
    pass


import torch
from rsl_rl.runners.on_policy_runner import OnPolicyRunner
from metasim.cfg.scenario import ScenarioCfg
from roboverse_learn.unitree_rl.utils import (
    export_policy_as_jit,
    get_args,
    get_export_jit_path,
    get_load_path,
    get_log_dir,
    get_class,
    make_robots,
)


def play(args):
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    _robots_name, _robots = make_robots(args)
    robots_name, robots = [_robots_name[0]], [_robots[0]]
    config_wrapper = get_class(args.task, "Cfg")
    task = config_wrapper(robots=robots)
    scenario = ScenarioCfg(
        task=task,
        decimation=task.decimation,
        robots=robots,
        num_envs=args.num_envs,
        sim=args.sim,
        headless=args.headless,
        cameras=[],
    )
    scenario.num_envs = 1
    scenario.task.commands.curriculum = False
    scenario.task.ppo_cfg.runner.resume = True
    scenario.task.random.friction.enabled = False
    scenario.task.random.mass.enabled = False
    scenario.task.random.push.enabled = False

    log_dir = get_log_dir(args, scenario)
    task_wrapper = get_class(args.task, "Task")
    env = task_wrapper(scenario)

    load_path = get_load_path(args, scenario)

    obs = env.get_observations()
    # load policy
    try:
        ppo_runner = OnPolicyRunner(
            env=env,
            train_cfg=env.train_cfg,
            device=env.device,
            log_dir=log_dir,
            args=args,
        )
    except Exception as e:
        ppo_runner = OnPolicyRunner(
            env=env,
            train_cfg=env.train_cfg,
            device=env.device,
            log_dir=log_dir,
            # args=args,
        )
    ppo_runner.load(load_path)
    policy = ppo_runner.get_inference_policy(device=env.device)

    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        export_jit_path = get_export_jit_path(args, scenario)
        export_policy_as_jit(ppo_runner.alg.actor_critic, export_jit_path)
        print("Exported policy as jit script to: ", export_jit_path)

    for i in range(1000):
        # set fixed command
        env.commands[:, 0] = 0.5
        env.commands[:, 1] = 0.0
        env.commands[:, 2] = 0.0
        env.commands[:, 3] = 0.0

        actions = policy(obs.detach())
        obs, _, _, _, _ = env.step(actions.detach())


if __name__ == "__main__":
    EXPORT_POLICY = False
    args = get_args()
    args.task = "humanoid_walking" if args.task is None else args.task
    args.robot = "g1_dex3" if args.task is None else args.robot
    args.load_run = "2025_0808_084631" if args.load_run is None else args.load_run
    args.checkpoint = 0 if args.checkpoint is None else args.checkpoint
    args.sim = "isaacgym" if args.sim is None else args.sim
    play(args)

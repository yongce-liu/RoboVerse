import rootutils

rootutils.setup_root(__file__, pythonpath=True)
try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import os

import torch
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

from metasim.cfg.scenario import ScenarioCfg
from roboverse_learn.unitree_rl.helper.utils import (
    PolicyExporterLSTM,
    export_policy_as_jit,
    get_args,
    get_class,
    get_export_jit_path,
    get_load_path,
    make_robots,
)


def play(args):
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    _robots_name, _robots = make_robots(args)
    robots_name, robots = [_robots_name[0]], [_robots[0]]
    config_wrapper = get_class(args.task, "Cfg")
    task = config_wrapper(robots=robots)
    if args.sim == "mujoco":
        task.sim_params.dt = 0.002
        task.decimation = 10
        task.__post_init__()
    scenario = ScenarioCfg(
        task=task,
        sim_params=task.sim_params,
        decimation=task.decimation,
        robots=robots,
        num_envs=args.num_envs,
        sim=args.sim,
        headless=args.headless,
        try_add_table=True,  # add a ground plane
        cameras=[],
    )
    scenario.num_envs = 1
    scenario.task.commands.curriculum = False
    scenario.task.ppo_cfg.runner.resume = True
    scenario.task.random.friction.enabled = False
    scenario.task.random.mass.enabled = False
    scenario.task.random.push.enabled = False
    scenario.task.noise.add_noise = False

    task_wrapper = get_class(args.task, "Task")
    env = task_wrapper(scenario)

    load_path = get_load_path(args, scenario)
    # Use the existing run directory as log_dir to avoid creating new output dirs during play
    log_dir = os.path.dirname(load_path)

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
    if args.jit_load:
        policy = torch.jit.load(load_path).to(env.device)
    else:
        ppo_runner.load(load_path)
        policy = ppo_runner.get_inference_policy(device=env.device)

    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        export_jit_path = get_export_jit_path(args, scenario)
        actor_critic = ppo_runner.alg.actor_critic
        if hasattr(actor_critic, "memory_a"):
            exporter = PolicyExporterLSTM(actor_critic)
            exporter.export(export_jit_path)
        else:
            export_policy_as_jit(actor_critic.actor, export_jit_path)
        print("Exported policy as jit script to: ", export_jit_path)

    # if args.reindex_actions:
    num_actions = env.num_actions
    reindex_actions_idx = env.env.handler.get_joint_reindex(obj_name=env.robot.name, inverse=False)
    print(f"Reindex actions idx: {reindex_actions_idx}")
    reverse_reindex_actions_idx = env.env.handler.get_joint_reindex(obj_name=env.robot.name, inverse=True)
    assert num_actions == len(reindex_actions_idx)
    print(f"Reverse reindex actions idx: {reverse_reindex_actions_idx}")

    for i in range(1000):
        # # set fixed command
        # env.commands[:, 0] = 0.5
        # env.commands[:, 1] = 0.0
        # env.commands[:, 2] = 0.0
        # env.commands[:, 3] = 0.0
        commands = torch.tensor([0.0, 0.0, 0.0], device=env.device)
        actions = policy(obs.detach()).detach()
        if args.reindex_actions:
            actions = actions[:, reindex_actions_idx]
        obs, _, _, _, _ = env.step(actions)
        obs[:, :3] = commands.unsqueeze(0)
        if args.reindex_actions:
            # set the command
            obs[:, 9 : 9 + num_actions] = obs[:, 9 : 9 + num_actions][:, reverse_reindex_actions_idx]
            obs[:, 9 + num_actions : 9 + num_actions * 2] = obs[:, 9 + num_actions : 9 + num_actions * 2][
                :, reverse_reindex_actions_idx
            ]
            obs[:, 9 + num_actions * 2 : 9 + num_actions * 3] = obs[:, 9 + num_actions * 2 : 9 + num_actions * 3][
                :, reverse_reindex_actions_idx
            ]


if __name__ == "__main__":
    EXPORT_POLICY = True
    args = get_args()
    play(args)

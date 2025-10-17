from __future__ import annotations

import os

import rootutils

rootutils.setup_root(__file__, pythonpath=True)

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import torch
from loguru import logger as log
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

from metasim.task.registry import get_task_class
from roboverse_learn.rl.unitree_rl.helper.utils import (
    get_args, get_load_path
)


def play(args):
    """Play/evaluate a trained policy for a packaged task (registry-based).

    Mirrors train_pack.py setup: uses task registry to construct the env and
    RSL-RL OnPolicyRunner to load and run the policy.
    """

    if not args.load_run:
        raise ValueError("Please provide --load_run pointing to a run dir or checkpoint file.")

    # Resolve task from registry (keep consistent with train_pack.py)
    # Current default packaged example: humanoid.g1_dof29.walk
    task_cls = get_task_class("unitree_rl.g1_dof29.walk")

    # Build scenario from the class default and override runtime params
    scenario = task_cls.scenario.update(
        simulator=args.sim,
        num_envs=args.num_envs,
        headless=args.headless,
        cameras=[],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = task_cls(scenario=scenario, device=device)

    # Make eval deterministic: disable curriculum, noise and random pushes
    try:
        env.cfg.commands.curriculum = False
        env.cfg.random.push.enabled = False
        env.cfg.noise.add_noise = False
    except Exception as e:
        log.warning(f"Unable to fully disable randomization/noise for eval: {e}")

    # Initialize runner and load the policy
    load_path = get_load_path(args, scenario)
    log_dir = os.path.dirname(load_path) if os.path.isfile(load_path) else load_path

    try:
        runner = OnPolicyRunner(
            env=env,
            train_cfg=env.train_cfg,
            device=env.device,
            log_dir=log_dir,
            args=args,
        )
    except Exception:
        runner = OnPolicyRunner(
            env=env,
            train_cfg=env.train_cfg,
            device=env.device,
            log_dir=log_dir,
        )

    if getattr(args, "jit_load", False):
        policy = torch.jit.load(load_path).to(env.device)
    else:
        runner.load(load_path)
        policy = runner.get_inference_policy(device=env.device)

    # Optionally reindex actions to simulator joint order
    num_actions = env.num_actions
    reindex_actions_idx = env.handler.get_joint_reindex(obj_name=env.robot.name, inverse=False)
    reverse_reindex_actions_idx = env.handler.get_joint_reindex(obj_name=env.robot.name, inverse=True)
    if args.reindex_actions:
        assert num_actions == len(reindex_actions_idx)
        log.info(f"Using reindexed actions for robot '{env.robot.name}'.")

    # Warm-up observation
    obs = env.get_observations()

    # Simple evaluation loop
    for _ in range(1000):
        # Set a fixed command: [lin_x, lin_y, ang_z, heading]
        env.commands[:, 0] = 0.5
        env.commands[:, 1] = 0.0
        env.commands[:, 2] = 1.0
        env.commands[:, 3] = 0.0

        with torch.no_grad():
            actions = policy(obs.detach())

        if args.reindex_actions:
            actions = actions[:, reindex_actions_idx]

        obs, _, _, _, _ = env.step(actions)

        # If we reindex actions, also align observation segments that include actions
        if args.reindex_actions:
            start = 9  # observation layout matches play.py usage
            A = num_actions
            obs[:, start : start + A] = obs[:, start : start + A][:, reverse_reindex_actions_idx]
            obs[:, start + A : start + 2 * A] = obs[:, start + A : start + 2 * A][:, reverse_reindex_actions_idx]
            obs[:, start + 2 * A : start + 3 * A] = obs[:, start + 2 * A : start + 3 * A][
                :, reverse_reindex_actions_idx
            ]


if __name__ == "__main__":
    args = get_args()
    play(args)

from typing import Sequence
import torch
from roboverse_pack.tasks.unitree_rl.base.types import EnvTypes

def lin_vel_cmd_levels(
    env: EnvTypes,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
) -> torch.Tensor:
    if env.common_step_counter % env.max_episode_steps == 0:
        command_term = env.cfg.commands
        ranges = command_term.ranges
        limit_ranges = command_term.limit_ranges

        reward_term_scales = env.reward_scales[reward_term_name][0]
        reward = torch.mean(env.episode_rewards[reward_term_name][env_ids]) / env.max_episode_steps

        if reward > reward_term_scales * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.lin_vel_x = torch.clamp(
                torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
                limit_ranges.lin_vel_x[0],
                limit_ranges.lin_vel_x[1],
            ).tolist()
            ranges.lin_vel_y = torch.clamp(
                torch.tensor(ranges.lin_vel_y, device=env.device) + delta_command,
                limit_ranges.lin_vel_y[0],
                limit_ranges.lin_vel_y[1],
            ).tolist()

    return torch.tensor(env.cfg.commands.ranges.lin_vel_x[1], device=env.device)

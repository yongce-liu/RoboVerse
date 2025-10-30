import torch

from metasim.types import TensorState
from metasim.utils.math import quat_from_euler_xyz, quat_apply, wrap_to_pi, sample_uniform


from roboverse_pack.tasks.unitree_rl.base.types import EnvTypes
from roboverse_learn.rl.unitree_rl.configs.cfg_base import BaseEnvCfg


def resample_commands(env: EnvTypes, env_states: TensorState = None):
    """Randomly select commands for some environments.

    Args:
        env_ids (List[int]): Environments ids for which new commands are needed.
    """
    cfg: BaseEnvCfg.Commands = env.cfg.commands
    if cfg.value is None:
        cfg.value = torch.zeros(
            size=(env.num_envs, cfg.num_commands),
            dtype=torch.float,
            device=env.device
        )
    env_ids = (env._episode_steps % int(cfg.resampling_time / env.step_dt) == 0).nonzero(as_tuple=False).flatten()
    if len(env_ids) == 0:
        return
    r = torch.empty(len(env_ids), device=env.device)

    cfg.value[env_ids, 0] = r.uniform_(*cfg.ranges.lin_vel_x)
    cfg.value[env_ids, 1] = r.uniform_(*cfg.ranges.lin_vel_y)
    cfg.value[env_ids, 2] = r.uniform_(*cfg.ranges.ang_vel_yaw)
    if cfg.heading_command:
        cfg.value[env_ids, 2] = r.uniform_(*cfg.ranges.heading)

    # low_cmd_mask = torch.norm(cfg.value[env_ids, :2], dim=1) < 0.1
    random_mask = r.uniform_(0.0, 1.0) <= cfg.rel_standing_envs
    final_env_ids = random_mask.nonzero(as_tuple=False).flatten()
    cfg.value[env_ids][final_env_ids, :] *= 0.0

    if cfg.heading_command:
        env_states = env.get_states() if env_states is None else env_states
        robot_state = env_states.robots[env.name]
        base_quat = robot_state.root_state[:, 3:7]
        forward = quat_apply(base_quat, env.forward_vec)  # quat:[w, x, y, z], forward:[x, y, z]
        heading = torch.atan2(forward[:, 1], forward[:, 0])
        cfg.value[:, 2] = torch.clip(0.5 * wrap_to_pi(cfg.value[:, 2] - heading), -1.0, 1.0)


def push_by_setting_velocity(env: EnvTypes,
                             env_states: TensorState,
                             interval_range_s: tuple|int=5.0,
                             velocity_range: list[list]=[[0]*3, [0]*3]):
    """Randomly set robot's root velocity to simulate a push."""
    if not env.cfg.domain_rand.push_robots:
        return
    env_ids = torch.arange(env.num_envs, device=env.device)
    # push_interval = torch_rand_float(interval_range_s[0], interval_range_s[1], (1,1), device=env.device) / env.step_dt
    push_interval = int((interval_range_s[0]+interval_range_s[1]) / env.step_dt)

    push_env_ids = env_ids[env._episode_steps[env_ids] % push_interval == 0]
    if len(push_env_ids) == 0:
        return

    velocity_range = torch.tensor(velocity_range, device=env.device)
    env_states.robots[env.name].root_state[push_env_ids, 7:10] += sample_uniform(velocity_range[0], velocity_range[1], (len(push_env_ids), 3), device=env.device)

    env.handler.set_states(env_states, push_env_ids.tolist())

import torch

from metasim.types import TensorState
from metasim.utils.math import quat_rotate_inverse

from roboverse_pack.tasks.unitree_rl.base.types import EnvTypes

def root_height_below_minimum(env: EnvTypes, env_states: TensorState, minimum_height: float) -> torch.Tensor:
    """Terminate when the asset's root height is below the minimum height.

    Note:
        This is currently only supported for flat terrains, i.e. the minimum height is in the world frame.
    """
    robot_state = env_states.robots[env.name]
    base_quat = robot_state.root_state[:, 3:7]
    return robot_state.root_state[:, 2] < minimum_height

def bad_orientation(env: EnvTypes, env_states: TensorState, limit_angle: float) -> torch.Tensor:
    """Terminate when the asset's orientation is too far from the desired orientation limits.

    This is computed by checking the angle between the projected gravity vector and the z-axis.
    """
    # extract the used quantities (to enable type-hinting)
    robot_state = env_states.robots[env.name]
    base_quat = robot_state.root_state[:, 3:7]
    projected_gravity = quat_rotate_inverse(base_quat, env.gravity_vec)
    return torch.acos(-projected_gravity[:, 2]).abs() > limit_angle

def time_out(env: EnvTypes, env_states: TensorState) -> torch.Tensor:
    """Terminate the episode when the episode length exceeds the maximum episode length."""
    return env._episode_steps >= env.max_episode_steps

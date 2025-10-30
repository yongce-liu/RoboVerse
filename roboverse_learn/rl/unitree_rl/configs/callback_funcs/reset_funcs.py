import torch
from copy import deepcopy

from metasim.types import TensorState
from metasim.utils.math import quat_from_euler_xyz
from metasim.utils.tensor_util import torch_rand_float

from roboverse_learn.rl.unitree_rl.helper.utils import torch_rand_float_tensor
from roboverse_pack.tasks.unitree_rl.base.types import EnvTypes


def random_root_state(env: EnvTypes, env_ids: torch.Tensor | list, pose_range: list[list]=[[0]*6, [0]*6], velocity_range: list[list]=[[0]*6, [0]*6]) -> torch.Tensor:
    if len(env_ids) == 0:
        return
    if not env.cfg.domain_rand.randomize_initial_state:
        env.handler.set_states(states=env.initial_env_states, env_ids=env_ids)
        return

    # pose_range = pose_range.detach().clone().requires_grad_(False).to(env.device)
    # velocity_range = velocity_range.detach().clone().requires_grad_(False).to(env.device)
    pose_range = torch.tensor(pose_range, device=env.device, requires_grad=False)
    velocity_range = torch.tensor(velocity_range, device=env.device, requires_grad=False)

    # weak copy
    random_initial_robot_states = env.setup_initial_env_states.robots[env.name]
    random_initial_robot_states.root_state[env_ids, :3] = torch_rand_float_tensor(
        pose_range[0, :3], pose_range[1, :3], (len(env_ids), 3), device=env.device
    )
    random_initial_robot_states.root_state[env_ids, 3:7] = quat_from_euler_xyz(
        roll=torch_rand_float(pose_range[0, 3], pose_range[1, 3], (len(env_ids),1), device=env.device).squeeze(-1),
        pitch=torch_rand_float(pose_range[0, 4], pose_range[1, 4], (len(env_ids,),1), device=env.device).squeeze(-1),
        yaw=torch_rand_float(pose_range[0, 5], pose_range[1, 5], (len(env_ids,),1), device=env.device).squeeze(-1),
    )
    random_initial_robot_states.root_state[env_ids, 7:10] = torch_rand_float_tensor(
        velocity_range[0, :3], velocity_range[1, :3], (len(env_ids), 3), device=env.device
    )
    random_initial_robot_states.root_state[env_ids, 10:13] = torch_rand_float_tensor(
        velocity_range[0, 3:], velocity_range[1, 3:], (len(env_ids), 3), device=env.device
    )

    random_initial_robot_states.root_state[env_ids] += env.initial_env_states.robots[env.name].root_state[env_ids]

    # env.handler.set_states(states=default_initial_env_states, env_ids=env_ids)


def reset_joints_by_scale(env: EnvTypes, env_ids: torch.Tensor | list, position_range: list|tuple=(1.0, 1.0), velocity_range: list|tuple=(1.0, 1.0)) -> torch.Tensor:
    if len(env_ids) == 0:
        return
    if not env.cfg.domain_rand.randomize_initial_state:
        env.handler.set_states(states=env.initial_env_states, env_ids=env_ids)
        return

    # weak copy
    random_initial_robot_states = env.setup_initial_env_states.robots[env.name]
    # joint position
    random_initial_robot_states.joint_pos[env_ids] = env.initial_env_states.robots[env.name].joint_pos[env_ids] * torch_rand_float(
        position_range[0], position_range[1], (len(env_ids), env.num_actions), device=env.device
    )
    # joint velocity
    random_initial_robot_states.joint_vel[env_ids] = env.initial_env_states.robots[env.name].joint_vel[env_ids] * torch_rand_float(
        velocity_range[0], velocity_range[1], (len(env_ids), env.num_actions), device=env.device
    )

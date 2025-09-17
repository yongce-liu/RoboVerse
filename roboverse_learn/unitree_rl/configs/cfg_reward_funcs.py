import torch
from metasim.types import TensorState
from roboverse_learn.unitree_rl.envs.env_legged_robot import LeggedRobotEnv
from roboverse_learn.unitree_rl.envs.env_humanoid import HumanoidEnv
from roboverse_learn.unitree_rl.envs import EnvTypes


def reward_lin_vel_z(env: EnvTypes) -> torch.Tensor:
    """Reward for z linear velocity."""
    return torch.square(env.base_lin_vel[:, 2])

def reward_ang_vel_xy(env: EnvTypes) -> torch.Tensor:
    return torch.sum(torch.square(env.base_ang_vel[:, :2]), dim=1)

def reward_orientation(env: EnvTypes) -> torch.Tensor:
    """
    Penalize deviation from flat base orientation.
    """
    return torch.sum(torch.square(env.projected_gravity[:, :2]), dim=1)

def reward_base_height(env: EnvTypes) -> torch.Tensor:
    # Penalize base height away from target
    states: TensorState = env.get_states()
    base_height = states.robots[env.name].root_state[:, 2]
    return torch.square(base_height - env.cfg.rewards.extras.base_height_target)

def reward_torques(env: EnvTypes) -> torch.Tensor:
    """
    Penalize high torques.
    """
    return torch.sum(torch.square(env.torques), dim=1)

def reward_dof_vel(env: EnvTypes) -> torch.Tensor:
    """
    Penalize high DOF velocities.
    """
    states: TensorState = env.get_states()
    return torch.sum(torch.square(states.robots[env.name].joint_vel), dim=1)

def reward_dof_acc(env: EnvTypes) -> torch.Tensor:
    """
    Penalize high DOF accelerations.
    """
    states: TensorState = env.get_states()
    return torch.sum(torch.square((env.history_buffer["joint_vel"][-1] - states.robots[env.name].joint_vel) / env.dt), dim=1)

def reward_action_rate(env: EnvTypes) -> torch.Tensor:
    """
    Penalize high action rate.
    """
    return torch.sum(torch.square(env.history_buffer["actions"][-1] - env.actions), dim=1)

def reward_collision(env: EnvTypes) -> torch.Tensor:
    """
    Penalize collisions.
    """
    return torch.sum(1.0 * (torch.norm(env.contact_forces[:, env.penalised_contact_indices, :], dim=-1) > 0.1),dim=1)

def reward_termination(env: EnvTypes) -> torch.Tensor:
    """
    Reward for termination, used to reset the environment.
    """
    return env.reset_buf * ~env.time_out_buf

def reward_dof_pos_limits(env: EnvTypes) -> torch.Tensor:
    """
    Penalize DOF positions that are out of limits.
    """
    states: TensorState = env.get_states()
    out_of_limits = -(states.robots[env.name].joint_pos - env.dof_pos_limits[:, 0]).clip(max=0.0)
    out_of_limits += (states.robots[env.name].joint_pos - env.dof_pos_limits[:, 1]).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)

'''
def reward_dof_vel_limits(env: EnvTypes) -> torch.Tensor:
    """
    Penalize high DOF velocities.
    """
    states: TensorState = env.get_states()
    return torch.sum(
        (
            torch.abs(states.robots[env.name].joint_vel)
            - env.dof_vel_limits * env.soft_dof_vel_limit
        ).clip(min=0.0, max=1.0),
        dim=1,
    )
'''

def reward_torque_limits(env: EnvTypes) -> torch.Tensor:
    """
    Penalize high torques.
    """
    return torch.sum((torch.abs(env.torques) - env.torque_limits * env.cfg.rewards.extras.soft_torque_limit).clip(min=0.0), dim=1)

def reward_tracking_lin_vel(env: EnvTypes) -> torch.Tensor:
    """
    Track linear velocity commands (xy axes).
    """
    lin_vel_diff = env.commands[:, :2] - env.base_lin_vel[:, :2]
    lin_vel_error = torch.sum(torch.square(lin_vel_diff), dim=1)
    return torch.exp(-lin_vel_error / env.cfg.rewards.extras.tracking_sigma)

def reward_tracking_ang_vel(env: EnvTypes) -> torch.Tensor:
    """
    Track angular velocity commands (yaw).
    """
    ang_vel_diff = env.commands[:, 2] - env.base_ang_vel[:, 2]
    ang_vel_error = torch.square(ang_vel_diff)
    return torch.exp(-ang_vel_error / env.cfg.rewards.extras.tracking_sigma)

'''
def reward_feet_air_time(env: EnvTypes) -> torch.Tensor:
    """
    Calculates the reward for feet air time.
    """
    air_time = env.feet_air_time
    return air_time.sum(dim=1) * (torch.norm(env.commands[:, :2], dim=1) > 0.1)
'''

'''
def reward_stumble(env: EnvTypes) -> torch.Tensor:
    """
    Penalize stumbling based on contact forces.
    """
    return torch.any(
        torch.norm(env.contact_forces"][:, env.feet_indices, :2], dim=2)
        > 5 * torch.abs(env.contact_forces"][:, env.feet_indices, 2]),
        dim=1,
    )
'''

'''
def reward_stand_still(env: EnvTypes) -> torch.Tensor:
    """
    Reward for standing still, penalizing deviation from default joint positions.
    """
    return torch.sum(torch.abs(states.robots[env.name].joint_pos - env.cfg.default_dof_pos), dim=1) * (
        torch.norm(env.command"][:, :2], dim=1) < 0.1
    )
'''

'''
def reward_feet_contact_forces(env: EnvTypes) -> torch.Tensor:
    """
    Penalize high contact forces on feet.
    """
    return torch.sum(
        (
            torch.norm(
                env.contact_forces"][:, env.feet_indices, :],
                dim=-1,
            )
            - env.cfg.reward_env.cfg.max_contact_force
        ).clip(min=0, max=400),
        dim=1,
    )
'''

def reward_contact(env: EnvTypes) -> torch.Tensor:
    contact = env.contact_forces[:, env.feet_indices, 2] > 1.0
    res = torch.logical_not(torch.logical_xor(contact, env.leg_phase))
    return res.sum(dim=1, dtype=torch.float32)

def reward_feet_swing_height(env: EnvTypes) -> torch.Tensor:
    states: TensorState = env.get_states()
    contact = torch.norm(env.contact_forces[:, env.feet_indices, :3], dim=2) > 1.0
    feet_state = states.robots[env.name].body_state[:, env.feet_indices, :]
    feet_pos = feet_state[:, :, :3]
    pos_error = torch.square(feet_pos[:, :, 2] - env.cfg.rewards.extras.target_feet_height) * ~contact
    return torch.sum(pos_error, dim=1)

def reward_alive(env: EnvTypes) -> torch.Tensor:
    # Reward for staying alive
    return 1.0

def reward_contact_no_vel(env: EnvTypes) -> torch.Tensor:
    # Penalize contact with no velocity
    states: TensorState = env.get_states()
    contact = torch.norm(env.contact_forces[:, env.feet_indices, :3], dim=2) > 1.0
    feet_state = states.robots[env.name].body_state[:, env.feet_indices, :]
    feet_vel = feet_state[:, :, 7:10]
    contact_feet_vel = feet_vel * contact.unsqueeze(-1)
    penalize = torch.square(contact_feet_vel[:, :, :3])
    return torch.sum(penalize, dim=(1, 2))

def reward_hip_pos(env: EnvTypes) -> torch.Tensor:
    states: TensorState = env.get_states()
    dof_pos = states.robots[env.name].joint_pos
    indices = torch.concat([env.left_yaw_roll_joint_indices, env.right_yaw_roll_joint_indices])
    dof_pos_hip = dof_pos[:, indices]
    return torch.sum(torch.square(dof_pos_hip), dim=1)

# ==========================h1 walking========================
def reward_joint_pos(env: EnvTypes) -> torch.Tensor:
    """
    Calculates the reward based on the difference between the current joint positions and the target joint positions.
    """
    joint_pos = states.robots[env.name].joint_pos
    pos_target = env.actions + env.default_dof_pos
    diff = joint_pos - pos_target
    r = torch.exp(-2 * torch.norm(diff, dim=1)) - 0.2 * torch.norm(diff, dim=1).clamp(0, 0.5)
    return r, torch.mean(torch.abs(diff), dim=1)


def reward_feet_distance(env: EnvTypes) -> torch.Tensor:
    base = states.robots[env.name]
    feet_y = base.body_state[:, env.feet_indices, 1]  # (B, 2)
    step_width = torch.abs(feet_y[:, 0] - feet_y[:, 1])  # (B,)

    # Double support gating
    contact = env.contact_forces[:, env.feet_indices, 2] > 5.0
    both_stance = torch.all(contact, dim=1)

    # Step width band
    sw_min = getattr(env.cfg.reward_cfg, "min_dist", 0.18)
    sw_max = getattr(env.cfg.reward_cfg, "max_dist", 0.38)
    k = 100.0
    d_min = torch.clamp(step_width - sw_min, -0.5, 0.0)
    d_max = torch.clamp(step_width - sw_max, 0.0, 0.5)
    band = (torch.exp(-torch.abs(d_min) * k) + torch.exp(-torch.abs(d_max) * k)) / 2.0

    # Gate 1: Relax when there's lateral command (weaken step width constraint when vy_cmd is large)
    vy_cmd = base.extra["command"][:, 1] if base.extra["command"].shape[1] > 1 else 0.0
    vy_gate = getattr(env.cfg.reward_cfg, "sw_gate_vy", 0.2)  # m/s
    gate_cmd = 1.0 - torch.clamp(torch.abs(vy_cmd) / vy_gate, 0.0, 1.0)  # small vy→1, large vy→0

    # Gate 2: Relax when DCM error is large (don't constrain step width when "recovery" is needed)
    y = states.robots[env.name].root_state[:, 1]
    vy = base.extra["base_lin_vel"][:, 1]
    z0 = getattr(env.cfg.reward_cfg, "base_height_target", 0.9)
    z0_t = torch.clamp(torch.as_tensor(z0, device=y.device, dtype=y.dtype), min=0.2)
    omega = torch.sqrt(torch.tensor(9.81, device=y.device, dtype=y.dtype) / z0_t)
    xi = y + vy / omega
    xi_ref = vy_cmd / omega
    dxi = torch.abs(xi - xi_ref)
    gate_dcm = torch.exp(-dxi * getattr(env.cfg.reward_cfg, "sw_dcm_relax", 8.0))  # large error→small gate

    gate = gate_cmd * gate_dcm  # Combined effect of both gates

    # Combination: only effective during double support; when gate is 0, degrades to 1.0 (neutral, doesn't affect other terms)
    raw = torch.where(both_stance, band, torch.ones_like(step_width))
    reward = gate * raw + (1 - gate) * torch.ones_like(step_width)

    return reward, step_width


def reward_knee_distance(env: EnvTypes) -> torch.Tensor:
    """
    Calculates the reward based on the distance between the knee of the humanoid.
    """
    knee_pos = states.robots[env.name].body_state[:, env.cfg.knee_indices, :2]
    knee_dist = torch.norm(knee_pos[:, 0, :] - knee_pos[:, 1, :], dim=1)
    fd = env.cfg.reward_env.cfg.min_dist
    max_df = env.cfg.reward_env.cfg.max_dist / 2
    d_min = torch.clamp(knee_dist - fd, -0.5, 0.0)
    d_max = torch.clamp(knee_dist - max_df, 0, 0.5)
    return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2, knee_dist


def reward_elbow_distance(env: EnvTypes) -> torch.Tensor:
    """
    Calculates the reward based on the distance between the elbow of the humanoid.
    """
    elbow_pos = states.robots[env.name].body_state[:, env.cfg.elbow_indices, :2]
    elbow_dist = torch.norm(elbow_pos[:, 0, :] - elbow_pos[:, 1, :], dim=1)
    fd = env.cfg.reward_env.cfg.min_dist
    max_df = env.cfg.reward_env.cfg.max_dist / 2
    d_min = torch.clamp(elbow_dist - fd, -0.5, 0.0)
    d_max = torch.clamp(elbow_dist - max_df, 0, 0.5)
    return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2, elbow_dist


def reward_foot_slip(env: EnvTypes) -> torch.Tensor:
    """
    Calculates the reward for minimizing foot slip.
    """
    contact = env.contact_forces[:, env.feet_indices, 2] > 5.0
    foot_speed_norm = torch.norm(states.robots[env.name].body_state[:, env.feet_indices, 10:12], dim=2)
    rew = torch.sqrt(foot_speed_norm)
    rew *= contact
    return torch.sum(rew, dim=1)


def reward_feet_contact_number(env: EnvTypes) -> torch.Tensor:
    """
    Reward based on feet contact matching gait phase.
    """
    contact = env.contact_forces[:, env.feet_indices, 2] > 5.0
    stance_mask = env.gait_phase
    reward = torch.where(contact == stance_mask, 1.0, -0.3)
    return torch.mean(reward, dim=1)


def reward_default_joint_pos(env: EnvTypes) -> torch.Tensor:
    """
    Keep joint positions close to defaults (penalize yaw/roll).
    """
    joint_diff = states.robots[env.name].joint_pos - env.cfg.default_dof_pos
    left_yaw_roll = joint_diff[:, env.cfg.left_yaw_roll_joint_indices]
    right_yaw_roll = joint_diff[:, env.cfg.right_yaw_roll_joint_indices]
    yaw_roll = torch.norm(left_yaw_roll, dim=1) + torch.norm(right_yaw_roll, dim=1)
    yaw_roll = torch.clamp(yaw_roll - 0.1, 0, 50)
    return torch.exp(-yaw_roll * 100) - 0.01 * torch.norm(joint_diff, dim=1)


def reward_upper_body_pos(env: EnvTypes) -> torch.Tensor:
    """
    Keep upper body joints close to default positions.
    """
    joint_diff = states.robots[env.name].joint_pos - env.cfg.default_dof_pos
    upper_body_diff = joint_diff[:, env.cfg.upper_body_joint_indices]  # start from torso
    upper_body_error = torch.mean(torch.abs(upper_body_diff), dim=1)
    return torch.exp(-4 * upper_body_error), upper_body_error


def reward_base_acc(env: EnvTypes) -> torch.Tensor:
    """
    Penalize base acceleration.
    """
    root_acc = env.last_root_vel - states.robots[env.name].root_state[:, 7:13]
    rew = torch.exp(-torch.norm(root_acc, dim=1) * 3)
    return rew


def reward_vel_mismatch_exp(env: EnvTypes) -> torch.Tensor:
    """
    Penalize velocity mismatch.
    """
    lin_mismatch = torch.exp(-torch.square(env.base_lin_vel[:, 2]) * 10)
    ang_mismatch = torch.exp(-torch.norm(env.base_ang_vel[:, :2], dim=1) * 5.0)
    return (lin_mismatch + ang_mismatch) / 2.0


def reward_track_vel_hard(env: EnvTypes) -> torch.Tensor:
    """
    Track linear and angular velocity commands.
    """
    lin_vel_error = torch.norm(
        env.command[:, :2] - env.base_lin_vel[:, :2],
        dim=1,
    )
    lin_vel_error_exp = torch.exp(-lin_vel_error * 10)
    ang_vel_error = torch.abs(
        env.command[:, 2] - env.base_ang_vel[:, 2]
    )
    ang_vel_error_exp = torch.exp(-ang_vel_error * 10)
    linear_error = 0.2 * (lin_vel_error + ang_vel_error)
    return (lin_vel_error_exp + ang_vel_error_exp) / 2.0 - linear_error


def reward_feet_clearance(env: EnvTypes) -> torch.Tensor:
    """
    Reward swing leg clearance.
    """
    return env.feet_clearance


def reward_low_speed(env: EnvTypes) -> torch.Tensor:
    """
    Penalize speed mismatch with command.
    """
    absolute_speed = torch.abs(env.base_lin_vel[:, 0])
    absolute_command = torch.abs(env.command[:, 0])
    speed_too_low = absolute_speed < 0.5 * absolute_command
    speed_too_high = absolute_speed > 1.2 * absolute_command
    speed_desired = ~(speed_too_low | speed_too_high)
    sign_mismatch = torch.sign(env.base_lin_vel[:, 0]) != torch.sign(
        env.command[:, 0]
    )
    reward = torch.zeros_like(env.base_lin_vel[:, 0])
    reward[speed_too_low] = -1.0
    reward[speed_too_high] = 0.0
    reward[speed_desired] = 1.2
    reward[sign_mismatch] = -2.0
    return reward * (env.command[:, 0].abs() > 0.1)


def reward_action_smoothness(env: EnvTypes) -> torch.Tensor:
    """
    Penalize jerk in actions.
    """
    term_1 = torch.sum(
        torch.square(env.last_actions - env.actions),
        dim=1,
    )
    term_2 = torch.sum(
        torch.square(
            env.actions
            + env.last_last_actions
            - 2 * env.last_actions
        ),
        dim=1,
    )
    term_3 = 0.05 * torch.sum(torch.abs(env.actions), dim=1)
    return term_1 + term_2 + term_3


def reward_waist_joint_stability(env: EnvTypes) -> torch.Tensor:
    """
    Reward for keeping waist joints (yaw, roll, pitch) stable and close to default positions.
    This directly penalizes waist joint deviations and velocities to prevent shaking.
    """
    joint_pos = states.robots[env.name].joint_pos
    joint_vel = states.robots[env.name].joint_vel

    waist_indices = env.cfg.waist_joint_indices

    # Get waist joint positions and velocities
    waist_pos = joint_pos[:, waist_indices]
    waist_vel = joint_vel[:, waist_indices]

    # Default waist positions (should be close to 0 for stability)
    waist_default = env.cfg.default_dof_pos[:, waist_indices]

    # Penalize deviation from default positions
    pos_error = torch.norm(waist_pos - waist_default, dim=1)
    pos_penalty = torch.exp(-pos_error * 20.0)

    # Penalize high waist joint velocities
    vel_error = torch.norm(waist_vel, dim=1)
    vel_penalty = torch.exp(-vel_error * 15.0)

    # Combine position and velocity penalties
    waist_stability_reward = 0.6 * pos_penalty + 0.4 * vel_penalty

    return waist_stability_reward

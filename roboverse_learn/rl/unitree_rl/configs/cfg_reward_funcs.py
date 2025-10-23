import torch

from metasim.types import TensorState
from metasim.utils.math import quat_apply, quat_rotate_inverse, wrap_to_pi

from roboverse_pack.tasks.unitree_rl.envs import EnvTypes


# ----------------------------- small utils for reward functions -----------------------------
def _quat_conj(q):  # (B,4) -> (B,4)
    # q = [w, x, y, z]
    return torch.stack([q[:, 0], -q[:, 1], -q[:, 2], -q[:, 3]], dim=1)

def _quat_rotate(q, v):  # q: (B,4); v: (B,N,3) -> (B,N,3)
    # rotate v by quaternion q (assumes q normalized)
    w, x, y, z = q[:, 0:1], q[:, 1:2], q[:, 2:3], q[:, 3:4]  # (B,1)
    q_vec = torch.cat([x, y, z], dim=1)                     # (B,3)
    # cross(q_vec, v)
    t = 2.0 * torch.cross(q_vec.unsqueeze(1).expand_as(v), v, dim=2)
    return v + w.unsqueeze(1) * t + torch.cross(q_vec.unsqueeze(1).expand_as(v), t, dim=2)

def _quat_rotate_inv(q, v):  # rotate v by q^{-1}
    return _quat_rotate(_quat_conj(q), v)

def _cmd_norm(base):
    # Handles [vx] or [vx, vy, wz]
    if base.extra["command"].ndim == 2:
        return torch.norm(base.extra["command"], dim=1)
    return torch.abs(base.extra["command"][:, 0])
# ----------------------------- small utils for reward functions -----------------------------

def reward_lin_vel_z(env: EnvTypes, env_states: TensorState) -> torch.Tensor:
    """Reward for z linear velocity."""
    robot_state = env_states.robots[env.name]
    base_quat = robot_state.root_state[:, 3:7]
    base_lin_vel = quat_rotate_inverse(base_quat, robot_state.root_state[:, 7:10])
    return torch.square(base_lin_vel[:, 2])

def reward_ang_vel_xy_tmp(env: EnvTypes, env_states: TensorState) -> torch.Tensor:
    robot_state = env_states.robots[env.name]
    base_quat = robot_state.root_state[:, 3:7]
    xy = torch.norm(quat_rotate_inverse(base_quat, robot_state.root_state[:, 10:13])[:, :2], dim=1)
    return torch.square(torch.clamp(xy - 0.25, min=0.0)) * 1.0

def reward_ang_vel_xy(env: EnvTypes, env_states: TensorState) -> torch.Tensor:
    robot_state = env_states.robots[env.name]
    base_quat = robot_state.root_state[:, 3:7]
    base_ang_vel = quat_rotate_inverse(base_quat, robot_state.root_state[:, 10:13])
    return torch.sum(torch.square(base_ang_vel[:, :2]), dim=1)

def reward_orientation_tmp(env: EnvTypes, env_states: TensorState) -> torch.Tensor:
    """
    Penalize deviation from flat base orientation.
    """
    quat_mismatch = torch.exp(
        -torch.sum(torch.abs(states.robots[robot_name].extra["base_euler_xyz"][:, :2]), dim=1) * 10
    )
    orientation = torch.exp(-torch.norm(states.robots[robot_name].extra["projected_gravity"][:, :2], dim=1) * 20)
    allow_roll_gate = _vy_gate(states.robots[robot_name], getattr(cfg.reward_cfg, "upright_gate_vy", 0.12))
    return (quat_mismatch + orientation) / 2.0 * allow_roll_gate

def reward_orientation(env: EnvTypes, env_states: TensorState) -> torch.Tensor:
    """
    Penalize deviation from flat base orientation.
    """
    robot_state = env_states.robots[env.name]
    base_quat = robot_state.root_state[:, 3:7]
    projected_gravity = quat_rotate_inverse(base_quat, env.gravity_vec)
    return torch.sum(torch.square(projected_gravity[:, :2]), dim=1)

def reward_base_height_tmp(env: EnvTypes, env_states: TensorState) -> torch.Tensor:
    base = states.robots[robot_name]
    z_root = base.root_state[:, 2]

    # Soft stance weights from contact forces (0..1), avoids hard switches
    F = torch.norm(base.extra["contact_forces"][:, cfg.feet_indices, :], dim=-1)  # (B, nfeet)
    w = torch.clamp((F - 20.0) / 40.0, 0.0, 1.0)  # tune 20/40 by your scale

    z_feet = states.robots[robot_name].body_state[:, cfg.feet_indices, 2]
    w_sum = w.sum(dim=1)
    z_foot_ref = (z_feet * w).sum(dim=1) / (w_sum + 1e-6)

    # fallback to min foot height if no clear stance
    z_foot_ref = torch.where(
        (w_sum > 0.2),
        z_foot_ref,
        z_feet.min(dim=1).values,
    )

    # clearance offset (was 0.05); keep small and configurable
    clearance = getattr(cfg.reward_cfg, "height_clearance", 0.03)
    h = z_root - (z_foot_ref - clearance)

    # target & shaping
    h_tgt = cfg.reward_cfg.base_height_target
    err = h - h_tgt

    # Huber-ish: deadband then squared
    db = getattr(cfg.reward_cfg, "height_deadband", 0.015)  # 1.5 cm
    err_db = torch.clamp(torch.abs(err) - db, min=0.0)
    sigma = getattr(cfg.reward_cfg, "base_height_sigma", 0.04)  # 4 cm scale
    return torch.exp(-(err_db**2) / (2 * sigma**2))

def reward_base_height(env: EnvTypes, env_states: TensorState) -> torch.Tensor:
    # Penalize base height away from target
    robot_state = env_states.robots[env.name]
    base_height = robot_state.root_state[:, 2]
    return torch.square(base_height - env.cfg.rewards.extras.base_height_target)

def reward_torques(env: EnvTypes, env_states: TensorState) -> torch.Tensor:
    """
    Penalize high torques.
    """
    return torch.sum(torch.square(env.torques), dim=1)

def reward_dof_vel(env: EnvTypes, env_states: TensorState) -> torch.Tensor:
    """
    Penalize high DOF velocities.
    """
    robot_state = env_states.robots[env.name]
    return torch.sum(torch.square(robot_state.joint_vel), dim=1)

def reward_dof_acc(env: EnvTypes, env_states: TensorState) -> torch.Tensor:
    """
    Penalize high DOF accelerations.
    """
    robot_state = env_states.robots[env.name]
    return torch.sum(torch.square((env.history_buffer["joint_vel"][-1] - robot_state.joint_vel) / env.dt), dim=1)

def reward_action_rate(env: EnvTypes, env_states: TensorState) -> torch.Tensor:
    """
    Penalize high action rate.
    """
    return torch.sum(torch.square(env.history_buffer["actions"][-1] - env.actions), dim=1)

def reward_collision(env: EnvTypes, env_states: TensorState) -> torch.Tensor:
    """
    Penalize collisions.
    """
    contact_forces = env_states.extras["contact_forces"][env.name]
    return torch.sum(1.0 * (torch.norm(contact_forces[:, env.penalised_contact_indices, :], dim=-1) > 0.1),dim=1)

def reward_termination(env: EnvTypes, env_states: TensorState) -> torch.Tensor:
    """
    Reward for termination, used to reset the environment.
    """
    return env.reset_buf * ~env.time_out_buf

def reward_dof_pos_limits(env: EnvTypes, env_states: TensorState) -> torch.Tensor:
    """
    Penalize DOF positions that are out of limits.
    """
    robot_state = env_states.robots[env.name]
    out_of_limits = -(robot_state.joint_pos - env.dof_pos_limits[:, 0]).clip(max=0.0)
    out_of_limits += (robot_state.joint_pos - env.dof_pos_limits[:, 1]).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)

def reward_dof_vel_limits(env: EnvTypes, env_states: TensorState) -> torch.Tensor:
    """
    Penalize high DOF velocities.
    """
    states: TensorState = env.get_states()
    return torch.sum(
        (
            torch.abs(states.robots[env.name].joint_vel)
            - env.dof_vel_limits * env.cfg.rewards.extras.soft_dof_vel_limit
        ).clip(min=0.0, max=1.0),
        dim=1,
    )

def reward_torque_limits(env: EnvTypes, env_states: TensorState) -> torch.Tensor:
    """
    Penalize high torques.
    """
    return torch.sum((torch.abs(env.torques) - env.torque_limits * env.cfg.rewards.extras.soft_torque_limit).clip(min=0.0), dim=1)

def reward_tracking_lin_vel(env: EnvTypes, env_states: TensorState) -> torch.Tensor:
    """
    Track linear velocity commands (xy axes).
    """
    robot_state = env_states.robots[env.name]
    base_quat = robot_state.root_state[:, 3:7]
    base_lin_vel = quat_rotate_inverse(base_quat, robot_state.root_state[:, 7:10])
    lin_vel_diff = env.commands[:, :2] - base_lin_vel[:, :2]
    lin_vel_error = torch.sum(torch.square(lin_vel_diff), dim=1)
    # FOR DOF 29-
    # return torch.exp(-lin_vel_error * cfg.reward_cfg.tracking_sigma), torch.mean(torch.abs(lin_vel_diff), dim=1)
    return torch.exp(-lin_vel_error / env.cfg.rewards.extras.tracking_sigma)

def reward_tracking_ang_vel(env: EnvTypes, env_states: TensorState) -> torch.Tensor:
    """
    Track angular velocity commands (yaw).
    """
    robot_state = env_states.robots[env.name]
    base_quat = robot_state.root_state[:, 3:7]
    base_ang_vel = quat_rotate_inverse(base_quat, robot_state.root_state[:, 10:13])
    ang_vel_diff = env.commands[:, 2] - base_ang_vel[:, 2]
    ang_vel_error = torch.square(ang_vel_diff)
    # FOR DOF 29-
    # return torch.exp(-ang_vel_error * cfg.reward_cfg.tracking_sigma), torch.abs(ang_vel_diff)~
    return torch.exp(-ang_vel_error / env.cfg.rewards.extras.tracking_sigma)

def reward_feet_air_time(env: EnvTypes, env_states: TensorState) -> torch.Tensor:
    """
    Calculates the reward for feet air time.
    """
    air_time = env.feet_air_time
    return air_time.sum(dim=1) * (torch.norm(env.commands[:, :2], dim=1) > 0.1)

def reward_stumble(env: EnvTypes, env_states: TensorState) -> torch.Tensor:
    """
    Penalize stumbling based on contact forces.
    """
    return torch.any(
        torch.norm(env.contact_forces[:, env.feet_indices, :2], dim=2)
        > 5 * torch.abs(env.contact_forces[:, env.feet_indices, 2]),
        dim=1,
    )

def reward_stand_still(env: EnvTypes, env_states: TensorState) -> torch.Tensor:
    """
    Reward for standing still, penalizing deviation from default joint positions.
    """
    return torch.sum(torch.abs(env_states.robots[env.name].joint_pos - env.cfg.default_dof_pos), dim=1) * (
        torch.norm(env.commands[:, :2], dim=1) < 0.1
    )

def reward_feet_contact_forces(env: EnvTypes, env_states: TensorState) -> torch.Tensor:
    """
    Penalize high contact forces on feet.
    """
    return torch.sum(
        (
            torch.norm(
                env.contact_forces[:, env.feet_indices, :],
                dim=-1,
            )
            - env.cfg.reward_env.cfg.max_contact_force
        ).clip(min=0, max=400),
        dim=1,
    )

def reward_contact(env: EnvTypes, env_states: TensorState) -> torch.Tensor:
    contact_forces = env_states.extras["contact_forces"][env.name]
    contact = contact_forces[:, env.feet_indices, 2] > 1.0

    """Add phase into states"""
    leg_phase = torch.zeros(size=(env.num_envs, len(env.feet_indices)), dtype=torch.bool, device=env.device)
    phase = env.get_phase()
    sin_pos = torch.sin(2 * torch.pi * phase)
    # left foot stance
    leg_phase[:, 0] = sin_pos >= 0
    # right foot stance
    leg_phase[:, 1] = sin_pos < 0
    # Double support phase
    leg_phase[torch.abs(sin_pos) < env.cfg.rewards.extras.all_feet_contact_time / 2.0] = True

    res = torch.sum(contact == leg_phase, dim=1, dtype=torch.float)
    return res

def reward_feet_swing_height(env: EnvTypes, env_states: TensorState) -> torch.Tensor:
    robot_state = env_states.robots[env.name]
    contact_forces = robot_state.extra["contact_forces"]
    contact = torch.norm(contact_forces[:, env.feet_indices, :3], dim=2) > 1.0
    feet_state = robot_state.body_state[:, env.feet_indices, :]
    feet_pos = feet_state[:, :, :3]
    pos_error = torch.square(feet_pos[:, :, 2] - env.cfg.rewards.extras.target_feet_height) * ~contact
    return torch.sum(pos_error, dim=1)

def reward_alive(env: EnvTypes, env_states: TensorState) -> torch.Tensor:
    # Reward for staying alive
    return 1.0

def reward_contact_no_vel(env: EnvTypes, env_states: TensorState) -> torch.Tensor:
    # Penalize contact with no velocity
    robot_state = env_states.robots[env.name]
    contact_forces = robot_state.extra["contact_forces"]
    contact = torch.norm(contact_forces[:, env.feet_indices, :3], dim=2) > 1.0
    feet_state = robot_state.body_state[:, env.feet_indices, :]
    feet_vel = feet_state[:, :, 7:10]
    contact_feet_vel = feet_vel * contact.unsqueeze(-1)
    penalize = torch.square(contact_feet_vel[:, :, :3])
    return torch.sum(penalize, dim=(1, 2))

def reward_hip_pos(env: EnvTypes, env_states: TensorState) -> torch.Tensor:
    robot_state = env_states.robots[env.name]
    dof_pos = robot_state.joint_pos
    indices = torch.concat([env.left_yaw_roll_joint_indices, env.right_yaw_roll_joint_indices])
    dof_pos_hip = dof_pos[:, indices]
    return torch.sum(torch.square(dof_pos_hip), dim=1)

'''
def _vy_gate(base, thresh=0.12):
    if base.extra["command"].shape[1] <= 1:
        return torch.ones_like(base.extra["command"][:, 0])
    return (torch.abs(base.extra["command"][:, 1]) < thresh).float()

def reward_hip_pos(states, robot_name, cfg):
    base = states.robots[robot_name]
    gate = _vy_gate(base, getattr(cfg.reward_cfg, "hip_pos_gate_vy", 0.12))
    dof_pos = base.joint_pos
    indices = torch.concat([cfg.left_yaw_roll_joint_indices, cfg.right_yaw_roll_joint_indices])
    dof_pos_hip = dof_pos[:, indices]
    return torch.sum(torch.square(dof_pos_hip), dim=1) * gate

'''

# ==========================h1 walking========================
def reward_joint_pos(env: EnvTypes, env_states: TensorState) -> torch.Tensor:
    """
    Calculates the reward based on the difference between the current joint positions and the target joint positions.
    """
    robot_state = env_states.robots[env.name]
    joint_pos = robot_state.joint_pos
    pos_target = env.actions + env.default_dof_pos
    diff = joint_pos - pos_target
    r = torch.exp(-2 * torch.norm(diff, dim=1)) - 0.2 * torch.norm(diff, dim=1).clamp(0, 0.5)
    return r

def reward_feet_distance(env: EnvTypes, env_states: TensorState) -> torch.Tensor:
    base = env_states.robots[env.name]
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
    y = base.root_state[:, 1]
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

def reward_knee_distance(env: EnvTypes, env_states: TensorState) -> torch.Tensor:
    """
    Calculates the reward based on the distance between the knee of the humanoid.
    """
    knee_pos = env_states.robots[env.name].body_state[:, env.cfg.knee_indices, :2]
    knee_dist = torch.norm(knee_pos[:, 0, :] - knee_pos[:, 1, :], dim=1)
    fd = env.cfg.reward_env.cfg.min_dist
    max_df = env.cfg.reward_env.cfg.max_dist / 2
    d_min = torch.clamp(knee_dist - fd, -0.5, 0.0)
    d_max = torch.clamp(knee_dist - max_df, 0, 0.5)
    return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2, knee_dist

def reward_elbow_distance(env: EnvTypes, env_states: TensorState) -> torch.Tensor:
    """
    Calculates the reward based on the distance between the elbow of the humanoid.
    """
    elbow_pos = env_states.robots[env.name].body_state[:, env.cfg.elbow_indices, :2]
    elbow_dist = torch.norm(elbow_pos[:, 0, :] - elbow_pos[:, 1, :], dim=1)
    fd = env.cfg.reward_env.cfg.min_dist
    max_df = env.cfg.reward_env.cfg.max_dist / 2
    d_min = torch.clamp(elbow_dist - fd, -0.5, 0.0)
    d_max = torch.clamp(elbow_dist - max_df, 0, 0.5)
    return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2, elbow_dist

def reward_foot_slip(env: EnvTypes, env_states: TensorState) -> torch.Tensor:
    """
    Calculates the reward for minimizing foot slip.
    """
    contact = env_states.extras["contact_forces"][env.name][:, env.feet_indices, 2] > 5.0
    foot_speed_norm = torch.norm(env_states.robots[env.name].body_state[:, env.feet_indices, 10:12], dim=2)
    rew = torch.sqrt(foot_speed_norm)
    rew *= contact
    return torch.sum(rew, dim=1)

def reward_feet_contact_number(env: EnvTypes, env_states: TensorState) -> torch.Tensor:
    """
    Reward based on feet contact matching gait phase.
    """
    contact = env.contact_forces[:, env.feet_indices, 2] > 5.0
    stance_mask = env.gait_phase
    reward = torch.where(contact == stance_mask, 1.0, -0.3)
    return torch.mean(reward, dim=1)

def reward_default_joint_pos(env: EnvTypes, env_states: TensorState) -> torch.Tensor:
    """
    Keep joint positions close to defaults (penalize yaw/roll).
    """
    joint_diff = env_states.robots[env.name].joint_pos - env.cfg.default_dof_pos
    left_yaw_roll = joint_diff[:, env.left_yaw_roll_joint_indices]
    right_yaw_roll = joint_diff[:, env.right_yaw_roll_joint_indices]
    yaw_roll = torch.norm(left_yaw_roll, dim=1) + torch.norm(right_yaw_roll, dim=1)
    yaw_roll = torch.clamp(yaw_roll - 0.1, 0, 50)
    return torch.exp(-yaw_roll * 100) - 0.01 * torch.norm(joint_diff, dim=1)

def reward_upper_body_pos(env: EnvTypes, env_states: TensorState) -> torch.Tensor:
    """
    Keep upper body joints close to default positions.
    """
    joint_diff = env_states.robots[env.name].joint_pos - env.cfg.default_dof_pos
    upper_body_diff = joint_diff[:, env.cfg.upper_body_joint_indices]  # start from torso
    upper_body_error = torch.mean(torch.abs(upper_body_diff), dim=1)
    return torch.exp(-4 * upper_body_error), upper_body_error

def reward_base_acc(env: EnvTypes, env_states: TensorState) -> torch.Tensor:
    """
    Penalize base acceleration.
    """
    root_acc = env.last_root_vel - env_states.robots[env.name].root_state[:, 7:13]
    rew = torch.exp(-torch.norm(root_acc, dim=1) * 3)
    return rew

def reward_vel_mismatch_exp(env: EnvTypes, env_states: TensorState) -> torch.Tensor:
    """
    Penalize velocity mismatch.
    """
    robot_state = env_states.robots[env.name]
    base_quat = robot_state.root_state[:, 3:7]
    base_lin_vel = quat_rotate_inverse(base_quat, robot_state.root_state[:, 7:10])
    base_ang_vel = quat_rotate_inverse(base_quat, robot_state.root_state[:, 10:13])
    lin_mismatch = torch.exp(-torch.square(base_lin_vel[:, 2]) * 10)
    ang_mismatch = torch.exp(-torch.norm(base_ang_vel[:, :2], dim=1) * 5.0)
    return (lin_mismatch + ang_mismatch) / 2.0

def reward_track_vel_hard(env: EnvTypes, env_states: TensorState) -> torch.Tensor:
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

def reward_feet_clearance(env: EnvTypes, env_states: TensorState) -> torch.Tensor:
    """
    Reward swing leg clearance.
    """
    return env.feet_clearance

def reward_low_speed(env: EnvTypes, env_states: TensorState) -> torch.Tensor:
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

def reward_action_smoothness(env: EnvTypes, env_states: TensorState) -> torch.Tensor:
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

def reward_waist_joint_stability(env: EnvTypes, env_states: TensorState) -> torch.Tensor:
    """
    Reward for keeping waist joints (yaw, roll, pitch) stable and close to default positions.
    This directly penalizes waist joint deviations and velocities to prevent shaking.
    """
    robot_state = env_states.robots[env.name]
    joint_pos = robot_state.joint_pos
    joint_vel = robot_state.joint_vel

    return torch.exp(-20 * (joint_pos[:, env.upper_body_joint_indices]-env.default_dof_pos[env.upper_body_joint_indices]).abs()).mean(dim=1)
    # waist_indices = env.waist_joint_indices[]

    # # Get waist joint positions and velocities
    # waist_pos = joint_pos[:, waist_indices]
    # waist_vel = joint_vel[:, waist_indices]

    # # Default waist positions (should be close to 0 for stability)
    # waist_default = env.default_dof_pos[waist_indices]

    # # Penalize deviation from default positions
    # pos_error = torch.norm(waist_pos - waist_default, dim=1)
    # pos_penalty = torch.exp(-pos_error * 20.0)

    # # Penalize high waist joint velocities
    # vel_error = torch.norm(waist_vel, dim=1)
    # vel_penalty = torch.exp(-vel_error * 15.0)

    # # Combine position and velocity penalties
    # waist_stability_reward = 0.6 * pos_penalty + 0.4 * vel_penalty

    # return waist_stability_reward

def reward_hip_upright_axis(env: EnvTypes, env_states: TensorState) -> torch.Tensor:
    """
    Reward for keeping hip/pelvis axis oriented upward (vertical).
    This penalizes hip tilting and rolling motions that cause shaking.

    Uses the pelvis/hip body orientation to ensure the local Z-axis stays aligned with world Z-axis.
    """
    # Get hip/pelvis body indices - typically the torso or pelvis link
    if hasattr(cfg, 'torso_indices') and len(cfg.torso_indices) > 0:
        hip_body_idx = cfg.torso_indices[0]  # Use first torso link as hip reference
    else:
        # Fallback to base body (root link) if no torso indices defined
        hip_body_idx = 0

    # Get body state for the hip/pelvis
    body_quat = states.robots[robot_name].body_state[:, hip_body_idx, 3:7]  # quaternion [w, x, y, z]

    # Convert quaternion to rotation matrix to get local Z-axis direction
    # Local Z-axis in world coordinates after rotation
    w, x, y, z = body_quat[:, 0], body_quat[:, 1], body_quat[:, 2], body_quat[:, 3]

    # Extract local Z-axis (3rd column of rotation matrix)
    # R[2,2] = 1 - 2*(x^2 + y^2) - this is the Z-component of local Z-axis in world frame
    local_z_world_z = 1 - 2 * (x**2 + y**2)

    # We want local Z-axis to be aligned with world Z-axis (pointing up)
    # Perfect alignment: local_z_world_z = 1, worst case: local_z_world_z = -1
    alignment_error = 1.0 - local_z_world_z  # Error ranges from 0 (perfect) to 2 (upside down)

    # Use exponential reward function - higher reward for better alignment
    hip_upright_reward = torch.exp(-alignment_error * 5.0)  # Scale factor 5.0 for sensitivity

    return hip_upright_reward

## ======================== Unitree Lab RL ========================
def reward_energy(env: EnvTypes, env_states: TensorState) -> torch.Tensor:
    """Sum |qdot|*|tau| across joints (\"energy\" usage)."""
    base = env_states.robots[env.name]
    qvel = torch.abs(base.joint_vel)
    tau  = torch.abs(base.joint_effort_target if base.joint_effort_target is not None else torch.zeros_like(qvel))
    return torch.sum(qvel * tau, dim=1)  # matches Unitree's energy()

def reward_stand_still_unitree(env: EnvTypes, env_states: TensorState) -> torch.Tensor:
    return reward_stand_still(env, env_states)  # uses your implementation

def reward_joint_position_penalty(env: EnvTypes, env_states: TensorState,
                                  stand_still_scale: float = 1.0,
                                  velocity_threshold: float = 0.05) -> torch.Tensor:
    """
    Penalize ||q - q_default||; scale it up when the robot is basically standing still
    (low command AND low base xy-speed), as in Unitree's joint_position_penalty().
    """
    base = env_states.robots[env.name]
    q     = base.joint_pos
    q_ref = env.default_dof_pos  # shape: (n_dof,) or (1,n_dof) broadcast OK
    err_l2 = torch.linalg.norm(q - q_ref, dim=1)

    cmd = _cmd_norm(base)
    body_vel_xy = torch.linalg.norm(base.extra["base_lin_vel"][:, :2], dim=1)  # world XY ok here

    idle = torch.logical_and(cmd <= 0.0, body_vel_xy <= velocity_threshold)
    return torch.where(idle, stand_still_scale * err_l2, err_l2)

# ----------------------------- base orientation -----------------------------
def reward_orientation_l2(env: EnvTypes, env_states: TensorState) -> torch.Tensor:
    """
    L2-squared kernel on cosine alignment between projected gravity in body frame and desired vector.
    normalized = 0.5 * cos + 0.5 ; return normalized**2. Matches Unitree's orientation_l2().
    """
    desired_gravity=(0.0, 0.0, 1.0)
    base = env_states.robots[env.name]
    robot_state = env_states.robots[env.name]
    base_quat = robot_state.root_state[:, 3:7]
    projected_gravity = quat_rotate_inverse(base_quat, env.gravity_vec)
    g_b = projected_gravity  # (B,3), gravity dir in base frame
    desired = torch.as_tensor(desired_gravity, device=g_b.device, dtype=g_b.dtype)
    cos_dist = torch.sum(g_b * desired, dim=-1)
    normalized = 0.5 * cos_dist + 0.5
    return torch.square(normalized)

def reward_upward(env: EnvTypes, env_states: TensorState) -> torch.Tensor:
    """
    Square of (1 - g_b.z) where g_b is gravity projected in base frame.
    Matches Unitree's upward(). Lower is better when gz->1.
    """
    gz = states.robots[robot_name].extra["projected_gravity"][:, 2]
    return torch.square(1.0 - gz)

# ----------------------------- feet rewards -----------------------------
def reward_feet_stumble(env: EnvTypes, env_states: TensorState) -> torch.Tensor:
    """
    Penalize \"hitting vertical surfaces\": any foot with ||F_xy|| > k * |F_z|.
    Unitree uses k=4 (your 'reward_stumble' used 5).
    """
    xy_over_z_ratio: float = 4.0
    F = states.robots[robot_name].extra["contact_forces"][:, cfg.feet_indices, :]  # (B,2,3)
    Fz = torch.abs(F[:, :, 2])
    Fxy = torch.linalg.norm(F[:, :, :2], dim=2)
    return torch.any(Fxy > xy_over_z_ratio * Fz, dim=1).float()

def reward_feet_height_body(env: EnvTypes, env_states: TensorState) -> torch.Tensor:
    """
    Swing feet height tracking (body frame) weighted by foot XY speed (body frame),
    gated by |command|>0.1 and scaled by clamp(-g_b.z, 0, 0.7)/0.7 (per Unitree).
    """
    tanh_mult: float = 3.0
    base = states.robots[robot_name]
    B, Nfeet = base.body_state.shape[0], len(cfg.feet_indices)

    # world positions/velocities
    foot_pos_w = base.body_state[:, cfg.feet_indices, 0:3]      # (B,2,3)
    foot_lin_w = base.body_state[:, cfg.feet_indices, 7:10]     # (B,2,3)
    root_pos_w = base.root_state[:, 0:3].unsqueeze(1)           # (B,1,3)
    root_quat  = base.root_state[:, 3:7]                        # (B,4) [w,x,y,z]

    # translate to root, then rotate into body frame
    rel_pos_w = foot_pos_w - root_pos_w
    pos_b = _quat_rotate_inv(root_quat, rel_pos_w)              # (B,2,3)
    vel_b = _quat_rotate_inv(root_quat, foot_lin_w)             # (B,2,3)

    foot_z_err2 = torch.square(pos_b[:, :, 2] - target_height)  # (B,2)
    vel_xy_tanh = torch.tanh(tanh_mult * torch.linalg.norm(vel_b[:, :, :2], dim=2))  # (B,2)

    reward = torch.sum(foot_z_err2 * vel_xy_tanh, dim=1)
    # gates/scales
    cmd_gate = (_cmd_norm(base) > 0.1).float()
    gz = base.extra["projected_gravity"][:, 2]
    grav_scale = torch.clamp(-gz, 0.0, 0.7) / 0.7
    return reward * cmd_gate * grav_scale

def reward_foot_clearance_exp(env: EnvTypes, env_states: TensorState) -> torch.Tensor:
    """
    Exponential swing clearance reward in world frame (Unitree's foot_clearance_reward).
    """
    std: float = 0.03
    tanh_mult: float = 3.0
    base = env_states.robots[env.name]
    foot_z = base.body_state[:, env.feet_indices, 2]
    foot_v_xy = torch.linalg.norm(base.body_state[:, env.feet_indices, 7:10][:, :, :2], dim=2)
    term = torch.square(foot_z - env.cfg.rewards.extras.target_feet_height) * torch.tanh(tanh_mult * foot_v_xy)
    return torch.exp(-torch.sum(term, dim=1) / std)

def reward_feet_too_near(env: EnvTypes, env_states: TensorState) -> torch.Tensor:
    """
    (threshold - distance_between_feet).clamp(min=0). Matches Unitree's feet_too_near().
    """
    threshold: float = 0.2
    base = states.robots[robot_name]
    feet_pos = base.body_state[:, cfg.feet_indices, 0:3]  # (B,2,3)
    dist = torch.linalg.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return torch.clamp(threshold - dist, min=0.0)

def reward_feet_contact_without_cmd(env: EnvTypes, env_states: TensorState) -> torch.Tensor:
    """
    Sum of foot-contacts when |cmd|<0.1 (\"stand without moving\").
    """
    base = states.robots[robot_name]
    is_contact = (base.extra["contact_forces"][:, cfg.feet_indices, 2] > 0.0)  # bool (B,2)
    reward = torch.sum(is_contact, dim=1).float()
    return reward * (_cmd_norm(base) < 0.1).float()

def reward_air_time_variance_penalty(env: EnvTypes, env_states: TensorState) -> torch.Tensor:
    """
    Var(air_time) + Var(contact_time) capped at 0.5s. If contact_time isn't available, use zeros.
    Mirrors Unitree's air_time_variance_penalty().
    """
    base = states.robots[robot_name]
    air = base.extra.get("feet_air_time", None)        # (B,2)
    con = base.extra.get("feet_contact_time", None)    # (B,2) optional

    if air is None:
        return torch.zeros(base.root_state.shape[0], device=base.root_state.device)

    air_c = torch.clamp(air, max=0.5)
    var_air = torch.var(air_c, dim=1, unbiased=False)

    if con is None:
        var_con = torch.zeros_like(var_air)
    else:
        con_c = torch.clamp(con, max=0.5)
        var_con = torch.var(con_c, dim=1, unbiased=False)

    return var_air + var_con

# ----------------------------- gait reward -----------------------------
def reward_feet_gait(env: EnvTypes, env_states: TensorState) -> torch.Tensor:
    """
    Compare desired stance (from leg phase = (global_phase + offset) % 1) to actual contact.
    If per-leg phase already exists at states.extra['leg_phase'] (shape BxNfeet), we use it.
    Otherwise we synthesize phases from a global phase in states.extra['global_phase'] if present.
    Mirrors Unitree's feet_gait() behavior.
    """
    gate_with_command: bool = False
    base = env_states.robots[env.name]
    contact = (env_states.extras["contact_forces"][env.name][:, env.feet_indices, 2] > 1.0)  # (B,2) bool

    is_stance = env._get_gait_phase()

    # XNOR: reward +1 when desired stance matches contact
    rew = torch.sum(contact == is_stance, dim=1, dtype=torch.float)

    if gate_with_command:
        rew *= (_cmd_norm(base) > 0.1).float()
    return rew

# ----------------------------- other rewards -----------------------------
def reward_joint_mirror(env: EnvTypes, env_states: TensorState) -> torch.Tensor:
    """
    Sum of squared joint position differences for mirror pairs.
    `mirror_joints` can be:
      - a list of (left_indices, right_indices) integer pairs, or
      - a list of (\"left_name\", \"right_name\") where cfg.joint_name_to_index maps names->idx.
    Matches Unitree's joint_mirror().
    """
    base = env_states.robots[env.name]
    q = base.joint_pos  # (B, ndof)

    # resolve names to indices if needed
    resolved = []
    for a, b in mirror_joints:
        if isinstance(a, str):
            a_idx = cfg.joint_name_to_index[a]
            b_idx = cfg.joint_name_to_index[b]
        else:
            a_idx, b_idx = a, b
        resolved.append((a_idx, b_idx))

    if len(resolved) == 0:
        return torch.zeros(q.shape[0], device=q.device)

    total = torch.zeros(q.shape[0], device=q.device)
    for a_idx, b_idx in resolved:
        total += torch.sum(torch.square(q[:, a_idx] - q[:, b_idx]), dim=-1)

    return total * (1.0 / len(resolved))

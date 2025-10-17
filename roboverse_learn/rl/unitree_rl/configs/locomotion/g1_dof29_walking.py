from __future__ import annotations

import torch

from metasim.utils import configclass

from roboverse_learn.rl.unitree_rl.configs.base_legged import BaseLeggedTaskCfg, LeggedRobotCfgPPO

@configclass
class G1Dof29WalkingCfgPPO(LeggedRobotCfgPPO):
    seed: int = 0

    algorithm = LeggedRobotCfgPPO.Algorithm(
        entropy_coef=0.001, learning_rate=1e-5, num_learning_epochs=2, gamma=0.994, lam=0.9
    )
    runner = LeggedRobotCfgPPO.Runner(
        num_steps_per_env=60, max_iterations=15001, save_interval=100, experiment_name="g1_dof29_walking"
    )


@configclass
class G1Dof29WalkingCfg(BaseLeggedTaskCfg):
    """Walking task configuration for Unitree G1 29DoF (no hands)."""

    task_name = "g1_dof29_walking"
    env_spacing: float = 1.0
    max_episode_length_s: int = 24
    control = BaseLeggedTaskCfg.ControlCfg(action_scale=0.25, action_offset=True, torque_limit_scale=0.85)

    # Initial state for the specific robot (no hand joints)
    init_states = [
        {
            "objects": {},
            "robots": {
                "g1_dof29": {
                    "pos": torch.tensor([0.0, 0.0, 0.8]),
                    "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    "dof_pos": {
                        # Hips & legs
                        "left_hip_pitch_joint": -0.1,
                        "left_hip_roll_joint": 0.0,
                        "left_hip_yaw_joint": 0.0,
                        "left_knee_joint": 0.3,
                        "left_ankle_pitch_joint": -0.2,
                        "left_ankle_roll_joint": 0.0,
                        "right_hip_pitch_joint": -0.1,
                        "right_hip_roll_joint": 0.0,
                        "right_hip_yaw_joint": 0.0,
                        "right_knee_joint": 0.3,
                        "right_ankle_pitch_joint": -0.2,
                        "right_ankle_roll_joint": 0.0,
                        # Waist
                        "waist_yaw_joint": 0.0,
                        "waist_roll_joint": 0.0,
                        "waist_pitch_joint": 0.0,
                        # Shoulders & arms
                        "left_shoulder_pitch_joint": 0.0,
                        "left_shoulder_roll_joint": 0.0,
                        "left_shoulder_yaw_joint": 0.0,
                        "left_elbow_joint": 0.0,
                        "left_wrist_roll_joint": 0.0,
                        "left_wrist_pitch_joint": 0.0,
                        "left_wrist_yaw_joint": 0.0,
                        "right_shoulder_pitch_joint": 0.0,
                        "right_shoulder_roll_joint": 0.0,
                        "right_shoulder_yaw_joint": 0.0,
                        "right_elbow_joint": 0.0,
                        "right_wrist_roll_joint": 0.0,
                        "right_wrist_pitch_joint": 0.0,
                        "right_wrist_yaw_joint": 0.0,
                    },
                }
            },
        }
    ]

    ppo_cfg = G1Dof29WalkingCfgPPO()

    frame_stack = 1
    c_frame_stack = 3

    reward_cfg = BaseLeggedTaskCfg.RewardCfg(base_height_target=0.76, tracking_sigma=1 / 0.25, max_contact_force=700)

    reward_weights: dict[str, float] = {
        # task tracking (mapped to your existing tracking funcs)
        "tracking_lin_vel": 1.0,      # from track_lin_vel_xy_yaw_frame_exp
        "tracking_ang_vel": 0.5,      # from track_ang_vel_z_exp
        "alive": 0.15,                # is_alive

        # base dynamics / effort
        "lin_vel_z": -2.0,            # lin_vel_z_l2
        "ang_vel_xy": -0.05,          # ang_vel_xy_l2
        "dof_vel": -0.001,            # joint_vel_l2
        "dof_acc": -2.5e-7,           # joint_acc_l2
        "action_rate": -0.05,         # action_rate_l2
        "dof_pos_limits": -5.0,       # joint_pos_limits
        "energy": -2e-5,              # energy

        # stability
        # "hip_upright_axis": 5.0,
        "waist_joint_stability": 2.0,  # waist_joint_stability

        # robot posture
        "orientation_l2": -5.0,       # flat_orientation_l2 -> orientation_l2 (your func name)
        "base_height_sq": -10.0,      # base_height_l2 -> base_height_sq (your L2 version)

        # feet / gait
        "feet_gait": 0.5,             # feet_gait
        "foot_slip": -0.2,            # feet_slide -> foot_slip (your equivalent)
        "foot_clearance_exp": 1.0,    # foot_clearance_reward -> foot_clearance_exp (your port)

        # other contacts
        "collision": -1.0,            # undesired_contacts -> collision (your penalised contacts)
    }

    def __post_init__(self):
        super().__post_init__()
        self.num_single_obs: int = self.commands.commands_dim + 9 + 3 * self.num_actions + 2
        self.num_observations: int = int(self.frame_stack * self.num_single_obs)
        self.single_num_privileged_obs: int = self.commands.commands_dim + 12 + 4 * self.num_actions + 14
        self.num_privileged_obs = int(self.c_frame_stack * self.single_num_privileged_obs)

from __future__ import annotations
from metasim.utils import configclass
from typing import Callable

@configclass
class BaseCfg:
    episode_length_s = 20.0
    num_obs_single = 0
    obs_len_history = 0 # number of past + current observations to include in the observation
    num_priv_obs_single = 0
    priv_obs_len_history = 0 # number of past + current privileged observations to include in the privileged observation
    class control:
        control_type = 'P' # P: position, V: velocity, T: torques
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        class scales:
            torque_limits: float = 1.0 # scale torque limits from urdf
            dof_pos_limits: float = 1.0 # scale dof pos limits from urdf

    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [-1.0, 1.0] # min max [m/s]
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class rewards:
        send_timeouts = True
        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        functions: list[Callable] | str = "roboverse_learn.unitree_rl.configs.cfg_reward_funcs"
        class scales:
            # termination = -0.0
            # tracking_lin_vel = 1.0
            # tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            # ang_vel_xy = -0.05
            # orientation = -0.
            # torques = -0.00001
            # dof_vel = -0.
            # dof_acc = -2.5e-7
            # base_height = -0.
            # feet_air_time =  1.0
            # collision = -1.
            # feet_stumble = -0.0
            # action_rate = -0.01
            # stand_still = -0.

        class extras:
            tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
            soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
            soft_dof_vel_limit = 1.
            soft_torque_limit = 1.
            base_height_target = 1.
            target_feet_height = 0.06
            feet_cycle_time = 0.8
            all_feet_contact_time = 0.05
            max_contact_force = 100. # forces above this value are penalized

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 100.

    class domain_rand:
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_base_mass = False
        added_mass_range = [-1., 1.]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.

    class noise:
        add_noise = True
        noise_level = 1.0 # scales other values
        class scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    class initial_states:
        objects = {}
        robots = {"g1_dof12":
                    {"pos": [0.0, 0.0, 0.8],
                     "rot": [1.0, 0.0, 0.0, 0.0],
                     "joint_pos": {
                            # Hips & legs
                            "left_hip_yaw_joint": 0.0,
                            "left_hip_roll_joint": 0.0,
                            "left_hip_pitch_joint": -0.1,
                            "left_knee_joint": 0.3,
                            "left_ankle_pitch_joint": -0.2,
                            "left_ankle_roll_joint": 0.0,
                            "right_hip_yaw_joint": 0.0,
                            "right_hip_roll_joint": 0.0,
                            "right_hip_pitch_joint": -0.1,
                            "right_knee_joint": 0.3,
                            "right_ankle_pitch_joint": -0.2,
                            "right_ankle_roll_joint": 0.0,
                        },
                    },
                }

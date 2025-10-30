from __future__ import annotations
from typing import Callable
from dataclasses import MISSING

from metasim.utils import configclass
from metasim.utils.configclass import class_to_dict
from roboverse_learn.rl.unitree_rl.configs.cfg_queries import ContactForces


@configclass
class CallbacksCfg:
    setup: dict = {}
    reset: dict = {} # func_name: (func(env, env_ids,**kwargs), kwargs)
    step: dict = {} # func_name: (func(env, env_states, **kwargs), kwargs)
    terminate: dict = {} # func_name: (func(env, env_states, **kwargs), kwargs)
    query: dict = {}


@configclass
class BaseEnvCfg:
    """
    The base class of environment configuration for legged robots.
    """
    episode_length_s = 20.0
    obs_len_history = 0 # number of past observations to include in the observation
    priv_obs_len_history = 0 # number of past privileged observations to include in the privileged observation
    '''
    env_spacing = 2.5
    sim_params = SimParamCfg(dt=0.005,
                            substeps=1,
                            num_threads=10,
                            solver_type=1,
                            num_position_iterations=4,
                            num_velocity_iterations=0,
                            contact_offset=0.01,
                            rest_offset=0.0,
                            bounce_threshold_velocity=0.5,
                            max_depenetration_velocity=1.0,
                            default_buffer_size_multiplier=5,
                            replace_cylinder_with_capsule=True,
                            friction_correlation_distance=0.025,
                            friction_offset_threshold=0.04)
    '''

    @configclass
    class Control:
        torque_limits_factor: float = 1.0 # scale torque limits from urdf
        soft_joint_pos_limit_factor: float = 1.0 # scale dof pos limits from urdf
        action_clip: float = 100.0
        action_scale = 0.25
        action_offset = True
        decimation = 4
    control = Control()

    @configclass
    class Commands:
        @configclass
        class Ranges:
            lin_vel_x = [-1.0, 1.0] # min max [m/s]
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-3.14, 3.14]

        num_commands = 3 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        rel_standing_envs: float = 0
        ranges = Ranges()
        limit_ranges = Ranges()
        resample: Callable = MISSING
        value: any = MISSING
    commands = Commands()

    @configclass
    class Curriculum:
        enabled = False
        funcs: dict[str, Callable] = MISSING

    curriculum = Curriculum()

    @configclass
    class Rewards:
        @configclass
        class Scales:
            pass

        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        functions: list[Callable] | str = "roboverse_learn.rl.unitree_rl.configs.callback_funcs.reward_funcs"
        scales = Scales()

    rewards = Rewards()

    @configclass
    class DomainRand:
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_base_mass = False
        added_mass_range = [-1., 1.]
        push_robots = True
        push_interval = int(15/0.02) # [s] average time between pushes
        max_push_vel_xy = 1.
        randomize_initial_state = False
        add_noise2obs = True
    domain_rand = DomainRand()

    class InitialStates:
        objects = {}
        robots = {
            "g1_dof12": {"pos": [0.0, 0.0, 0.78]},
            "g1_dof23": {"pos": [0.0, 0.0, 0.78]},
            "g1_dof29_dex3": {"pos": [0.0, 0.0, 0.78]},
            "g1_dof29": {"pos": [0.0, 0.0, 0.78],
                         "default_joint_pos":{"left_hip_pitch_joint": -0.1,
                                              "right_hip_pitch_joint": -0.1,
                                              ".*_knee_joint": 0.3,
                                              ".*_ankle_pitch_joint": -0.2,
                                              ".*_shoulder_pitch_joint": 0.3,
                                              "left_shoulder_roll_joint": 0.25,
                                              "right_shoulder_roll_joint": -0.25,
                                              ".*_elbow_joint": 0.97,
                                              "left_wrist_roll_joint": 0.15,
                                              "right_wrist_roll_joint": -0.15},
                        }
            }
    initial_states = InitialStates()

    callbacks: CallbacksCfg | dict | None = CallbacksCfg(
        query={"contact_forces": ContactForces(history_length=3)})

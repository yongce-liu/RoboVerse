from __future__ import annotations
from typing import Callable, Literal
from dataclasses import MISSING

from metasim.utils import configclass
from metasim.utils.configclass import class_to_dict
from roboverse_learn.rl.unitree_rl.configs.cfg_queries import ContactForces


@configclass
class CallbacksCfg:
    setup: dict = {}
    reset: dict = {}  # func_name: (func(env, env_ids,**kwargs), kwargs)
    step: dict = {}  # func_name: (func(env, env_states, **kwargs), kwargs)
    terminate: dict = {}  # func_name: (func(env, env_states, **kwargs), kwargs)
    query: dict = {}


@configclass
class BaseEnvCfg:
    """
    The base class of environment configuration for legged robots.
    """

    episode_length_s = 20.0
    obs_len_history = 0  # number of past observations to include in the observation
    priv_obs_len_history = 0  # number of past privileged observations to include in the privileged observation

    @configclass
    class Control:
        torque_limits_factor: float = 1.0  # scale torque limits from urdf
        soft_joint_pos_limit_factor: float = 1.0  # scale dof pos limits from urdf
        action_clip: float = 100.0
        action_scale = 0.25
        action_offset = True
        decimation = 4

    control = Control()

    @configclass
    class Commands:
        @configclass
        class Ranges:
            lin_vel_x = [-1.0, 1.0]  # min max [m/s]
            lin_vel_y = [-1.0, 1.0]  # min max [m/s]
            ang_vel_yaw = [-1, 1]  # min max [rad/s]
            heading = [-3.14, 3.14]

        num_commands = 3  # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10.0  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error
        rel_standing_envs: float = 0
        ranges = Ranges()
        ranges_tensor = None
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

        only_positive_rewards = False  # if true negative total rewards are clipped at zero (avoids early termination problems)
        functions: list[Callable] | str = (
            "roboverse_learn.rl.unitree_rl.configs.callback_funcs.reward_funcs"
        )
        scales = Scales()

    rewards = Rewards()

    class InitialStates:
        objects = {}
        robots = {
            "g1_dof12": {"pos": [0.0, 0.0, 0.78]},
            "g1_dof23": {"pos": [0.0, 0.0, 0.78]},
            "g1_dof29_dex3": {"pos": [0.0, 0.0, 0.78]},
            "g1_dof29": {
                "pos": [0.0, 0.0, 0.78],
                "default_joint_pos": {
                    "left_hip_pitch_joint": -0.1,
                    "right_hip_pitch_joint": -0.1,
                    ".*_knee_joint": 0.3,
                    ".*_ankle_pitch_joint": -0.2,
                    ".*_shoulder_pitch_joint": 0.3,
                    "left_shoulder_roll_joint": 0.25,
                    "right_shoulder_roll_joint": -0.25,
                    ".*_elbow_joint": 0.97,
                    "left_wrist_roll_joint": 0.15,
                    "right_wrist_roll_joint": -0.15,
                },
            },
        }

    initial_states = InitialStates()

    callbacks_setup: dict[str, tuple[Callable, dict] | Callable] = MISSING
    # func_name: (func(env, env_ids,**kwargs), kwargs)
    callbacks_reset: dict[str, tuple[Callable, dict] | Callable] = MISSING
    # func_name: (func(env, env_states, **kwargs), kwargs)
    callbacks_step: dict[str, tuple[Callable, dict] | Callable] = MISSING
    # func_name: (func(env, env_states, **kwargs), kwargs)
    callbacks_terminate: dict[str, tuple[Callable, dict] | Callable] = MISSING
    callbacks_query: dict[str, tuple[Callable, dict] | Callable] = MISSING

    def __post_init__(self):
        self.callbacks = CallbacksCfg()
        self.callbacks.query = self.callbacks_query
        self.callbacks.terminate = self.callbacks_terminate
        self.callbacks.setup = self.callbacks_setup
        self.callbacks.reset = self.callbacks_reset
        self.callbacks.step = self.callbacks_step

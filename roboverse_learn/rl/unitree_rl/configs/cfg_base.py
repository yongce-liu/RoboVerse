from __future__ import annotations
from typing import Callable, Literal
from dataclasses import MISSING

from metasim.utils import configclass
from metasim.scenario.simulator_params import SimParamCfg

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
        @configclass
        class Scales:
            torque_limits: float = 1.0 # scale torque limits from urdf
            dof_pos_limits: float = 1.0 # scale dof pos limits from urdf

        control_type = 'P' # P: position, V: velocity, T: torques
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        scales = Scales()
    control = Control()

    @configclass
    class Commands:
        @configclass
        class Ranges:
            lin_vel_x = [-1.0, 1.0] # min max [m/s]
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-3.14, 3.14]

        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        ranges = Ranges()
    commands = Commands()

    @configclass
    class Rewards:
        @configclass
        class Scales:
            termination = 0.0
            tracking_lin_vel = 0.0
            tracking_ang_vel = 0.0
            lin_vel_z = 0.0
            ang_vel_xy = 0.0
            orientation = 0.0
            torques = 0.0
            dof_vel = 0.0
            dof_acc = 0.0
            base_height = 0.0
            feet_air_time = 0.0
            collision = 0.0
            feet_stumble = 0.0
            action_rate = 0.0
            stand_still = 0.0
        @configclass
        class Extras:
            tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
            soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
            soft_dof_vel_limit = 1.
            soft_torque_limit = 1.
            base_height_target = 1.
            target_feet_height = 0.08
            feet_cycle_time = 0.8
            all_feet_contact_time = 0.05
            max_contact_force = 100. # forces above this value are penalized

        send_timeouts = True
        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        functions: list[Callable] | str = "roboverse_learn.rl.unitree_rl.configs.cfg_reward_funcs"
        scales = Scales()
        extras = Extras()
    rewards = Rewards()

    @configclass
    class Normalization:
        @configclass
        class ObsScales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0

        clip_observations = 100.
        clip_actions = 100.
        obs_scales = ObsScales()
    normalization = Normalization()

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
    domain_rand = DomainRand()

    @configclass
    class Noise:
        class Scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            # height_measurements = 0.1

        add_noise = True
        noise_level = 1.0 # scales other values
        scales = Scales()
    noise = Noise()

    class InitialStates:
        objects = {}
        robots = {
            "g1_dof12": {"pos": [0.0, 0.0, 0.8]},
            "g1_dof23": {"pos": [0.0, 0.0, 0.8]},
            "g1_dof29_dex3": {"pos": [0.0, 0.0, 0.8]},
                }
    initial_states = InitialStates()

'''
@configclass
class RslRlTrainCfg:
    """
    policy training cfg
    """
    @configclass
    class Policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        rnn_type: str = MISSING
        rnn_hidden_size: int = MISSING
        rnn_num_layers: int = MISSING

    @configclass
    class Algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3 #5.e-4
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

    @configclass
    class Runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24 # per iteration
        # max_iterations = 20001 # number of policy updates

        # logging
        save_interval = 50 # check for potential saves every this many iterations
        # experiment_name = 'test'
        # run_name = ''
        # # load and resume
        # resume = False
        # load_run = -1 # -1 = last run
        # checkpoint = -1 # -1 = last saved model
        # resume_path = None # updated from load_run and chkpt

    runner_class_name = 'OnPolicyRunner'
    # construct the object
    policy = Policy()
    algorithm = Algorithm()
    runner = Runner()
'''

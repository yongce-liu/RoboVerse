from __future__ import annotations
from typing import Callable
from metasim.utils import configclass
from metasim.scenario.simulator_params import SimParamCfg

@configclass
class BaseEnvCfg:
    episode_length_s = 20.0
    num_obs_single = 0
    obs_len_history = 0 # number of past + current observations to include in the observation
    num_priv_obs_single = 0
    priv_obs_len_history = 0 # number of past + current privileged observations to include in the privileged observation
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
            target_feet_height = 0.08
            feet_cycle_time = 0.7
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
        randomize_initial_state = True

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

class RslTrainCfg:
    seed = 1
    runner_class_name = 'OnPolicyRunner'
    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1

    class algorithm:
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

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24 # per iteration
        max_iterations = 1500 # number of policy updates

        # logging
        save_interval = 50 # check for potential saves every this many iterations
        experiment_name = 'test'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt

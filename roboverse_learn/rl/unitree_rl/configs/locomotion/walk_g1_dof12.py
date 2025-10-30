import math
from metasim.utils import configclass
from metasim.utils.configclass import class_to_dict
from roboverse_learn.rl.unitree_rl.configs.cfg_base import BaseEnvCfg
from roboverse_learn.rl.unitree_rl.configs.algorithm import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg, RslRlPpoActorCriticRecurrentCfg


@configclass
class WalkG1Dof12EnvCfg(BaseEnvCfg):
    episode_length_s = 24.0
    obs_len_history = 1
    priv_obs_len_history = 1

    control = BaseEnvCfg.Control(action_scale = 0.25,
                                 soft_joint_pos_limit_factor=0.98)

    @configclass
    class RewardsScales:
        track_lin_vel_xy = (1.0, {"std": math.sqrt(0.25)})
        track_ang_vel_z = (0.5, {"std": math.sqrt(0.25)})
        lin_vel_z = -2.0
        ang_vel_xy = -0.05
        flat_orientation = -1.0
        base_height = (-10.0, {"target_height": 0.78})
        joint_acc = -2.5e-7
        joint_vel = -0.001
        action_rate = -0.05
        joint_pos_limits = -5.0
        is_alive = 0.15
        joint_deviation_legs = -1.0
        feet_slide = -0.2
        # feet_swing_height = -20.0
        feet_clearance = (1.0, {"std": 0.05,
                                "tanh_mult": 2.0,
                                "target_height": 0.1,})
        # contact = 0.18
        feet_gait = (0.18, {"period": 0.8,
                        "offset": [0.0, 0.5],
                        "threshold": 0.55})
        energy = -0.00001
        ########################

    rewards = BaseEnvCfg.Rewards(
        scales = RewardsScales(),
        only_positive_rewards=True
    )

    @configclass
    class DomainRand:
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_base_mass = False
        added_mass_range = [-1.0, 1.0]
        push_robots = True
        push_interval = int(15 / 0.02)  # [s] average time between pushes
        max_push_vel_xy = 1.0
        randomize_initial_state = False
        add_noise2obs = True

    domain_rand = DomainRand(
        randomize_friction = True,
        friction_range = [0.1, 1.25],
        randomize_base_mass = True,
        added_mass_range = [-1., 3.],
        push_robots = True,
        push_interval = int(5/0.02),
        max_push_vel_xy = 0.5,
        randomize_initial_state = True
    )

    @configclass
    class Normalization:
        @configclass
        class ObsScales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
            quat = 1.0

        clip_observations = 100.
        clip_actions = 100.
        obs_scales = ObsScales()
    normalization = Normalization()

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

@configclass
class WalkG1Dof12RslRlTrainCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 100
    experiment_name = ""  # same as task name
    empirical_normalization = False
    policy = RslRlPpoActorCriticRecurrentCfg(
        init_noise_std = 0.8,
        actor_hidden_dims = [32],
        critic_hidden_dims = [32],
        activation = 'elu',
        rnn_type = 'lstm',
        rnn_hidden_dim = 64,
        rnn_num_layers = 1,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef = 1.0,
        use_clipped_value_loss = True,
        clip_param = 0.2,
        entropy_coef = 0.01,
        num_learning_epochs = 5,
        num_mini_batches = 4, # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3, #5.e-4
        schedule = 'adaptive', # could be adaptive, fixed
        gamma = 0.99,
        lam = 0.95,
        desired_kl = 0.01,
        max_grad_norm = 1.
    )

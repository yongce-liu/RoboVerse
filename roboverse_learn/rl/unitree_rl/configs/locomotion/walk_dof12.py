from metasim.utils import configclass
from roboverse_learn.rl.unitree_rl.configs.cfg_base import BaseEnvCfg
from roboverse_learn.rl.unitree_rl.third_party.isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg, RslRlPpoActorCriticRecurrentCfg


@configclass
class WalkDof12RslRlTrainCfg(RslRlOnPolicyRunnerCfg):
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


@configclass
class WalkDof12EnvCfg(BaseEnvCfg):
    obs_len_history = 0
    priv_obs_len_history = 0

    domain_rand = BaseEnvCfg.DomainRand(
        randomize_friction = True,
        friction_range = [0.1, 1.25],
        randomize_base_mass = True,
        added_mass_range = [-1., 3.],
        push_robots = True,
        push_interval = int(5/0.02),
        max_push_vel_xy = 1.5,
        randomize_initial_state = False
    )

    control = BaseEnvCfg.Control(action_scale = 0.25)

    @configclass
    class RewardsScales(BaseEnvCfg.Rewards.Scales):
        tracking_lin_vel = 1.0
        tracking_ang_vel = 0.5
        lin_vel_z = -2.0
        ang_vel_xy = -0.05
        orientation = -1.0
        base_height = -10.0
        dof_acc = -2.5e-7
        dof_vel = -1e-3
        action_rate = -0.01
        dof_pos_limits = -5.0
        alive = 0.15
        hip_pos = -1.0
        contact_no_vel = -0.2
        feet_swing_height = -20.0
        contact = 0.18
        torques = -0.00001
    @configclass
    class RewardExtras(BaseEnvCfg.Rewards.Extras):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.78
        feet_cycle_time = 0.8

    rewards = BaseEnvCfg.Rewards(
        scales = RewardsScales(),
        extras = RewardExtras()
    )

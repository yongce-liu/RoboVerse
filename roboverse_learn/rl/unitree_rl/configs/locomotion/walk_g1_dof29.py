from typing import Callable

from metasim.utils import configclass

from roboverse_learn.rl.unitree_rl.configs.cfg_base import BaseEnvCfg
from roboverse_learn.rl.unitree_rl.third_party.isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

@configclass
class WalkG1Dof29EnvCfg(BaseEnvCfg):
    """
    Environment configuration for humanoid walking task.
    """
    obs_len_history = 0
    priv_obs_len_history = 2
    episode_length_s = 24.0

    control = BaseEnvCfg.Control(action_scale = 0.25) # torque_limit_scale=0.85
    noise = BaseEnvCfg.Noise(add_noise=True)  # disable noise by default

    @configclass
    class RewardsScales:
        # task tracking (mapped to your existing tracking funcs)
        tracking_lin_vel = 1.0      # from track_lin_vel_xy_yaw_frame_exp
        tracking_ang_vel = 0.5      # from track_ang_vel_z_exp
        alive = 0.15                # is_alive

        # base dynamics / effort
        lin_vel_z = -2.0            # lin_vel_z_l2
        ang_vel_xy_tmp = -0.05          # ang_vel_xy_l2
        dof_vel = -0.001            # joint_vel_l2
        dof_acc = -2.5e-7           # joint_acc_l2
        action_rate = -0.05         # action_rate_l2
        dof_pos_limits = -5.0       # joint_pos_limits
        energy = -2e-5              # energy

        # stability
        # hip_upright_axis = 5.0,
        waist_joint_stability = 2.0  # waist_joint_stability

        # robot posture
        orientation_l2 = -5.0       # flat_orientation_l2 -> orientation_l2 (your func name)
        base_height = -10.0      # base_height_l2 -> base_height_sq (your L2 version)

        # feet / gait
        feet_gait = 0.5             # feet_gait
        foot_slip = -0.2            # feet_slide -> foot_slip (your equivalent)
        foot_clearance_exp = 1.0    # foot_clearance_reward -> foot_clearance_exp (your port)

        # other contacts
        collision = -1.0            # undesired_contacts -> collision (your penalised contacts)

    @configclass
    # class RewardExtras(BaseEnvCfg.Rewards.Extras):
    #     base_height_target=0.76
    #     tracking_sigma=0.25
    #     max_contact_force=700
    #     feet_cycle_time=0.64
    #     target_feet_height=0.06
    #     all_feet_contact_time=0.05
    #     soft_dof_pos_limit=0.9
    class RewardExtras(BaseEnvCfg.Rewards.Extras):
        base_height_target=0.76
        tracking_sigma=0.25
        max_contact_force=700
        feet_cycle_time=0.7
        target_feet_height=0.06
        all_feet_contact_time=0.05
        soft_dof_pos_limit=0.9

    rewards = BaseEnvCfg.Rewards(
        scales = RewardsScales(),
        extras = RewardExtras()
    )

@configclass
class WalkG1Dof29EnvRslRlTrainCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 60
    max_iterations = 15001
    save_interval = 100
    experiment_name = "walk_g1_dof29"  # same as task name
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[768, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=2,
        num_mini_batches=4,
        learning_rate=1e-5,
        schedule="adaptive",
        gamma=0.994,
        lam=0.9,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

from typing import Callable
import math

from metasim.utils import configclass

from roboverse_learn.rl.unitree_rl.configs.cfg_base import BaseEnvCfg, CallbacksCfg
from roboverse_learn.rl.unitree_rl.configs.algorithm import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from roboverse_learn.rl.unitree_rl.helper.curriculum_utils import lin_vel_cmd_levels
from roboverse_learn.rl.unitree_rl.configs.cfg_queries import ContactForces
from roboverse_learn.rl.unitree_rl.configs.cfg_randomizers import MaterialRandomizer


@configclass
class WalkG1Dof29EnvCfg(BaseEnvCfg):
    """
    Environment configuration for humanoid walking task.
    """
    obs_len_history = 5
    priv_obs_len_history = 5
    episode_length_s = 20.0

    control = BaseEnvCfg.Control(action_scale = 0.25,
                                 dof_pos_limits_factor=0.9)

    @configclass
    class RewardsScales:
        track_lin_vel_xy = (1.0, {"std": math.sqrt(0.25)})
        track_ang_vel_z = (0.5, {"std": math.sqrt(0.25)})
        is_alive = 0.15
        lin_vel_z = -2.0
        ang_vel_xy = -0.05
        joint_vel = -0.001
        joint_acc = -2.5e-7
        action_rate = -0.05
        joint_pos_limits = -5.0
        energy = -2e-5
        joint_deviation_arms = -0.1
        joint_deviation_waists = -1.0
        joint_deviation_legs = -1.0
        flat_orientation = -5.0
        base_height = (-10.0, {"target_height": 0.78})
        feet_gait = (0.5, {"period": 0.8,
                            "offset": [0.0, 0.5],
                            "threshold": 0.55})
        feet_slide = -0.2
        feet_clearance = (1.0, {"std": 0.05,
                                "tanh_mult": 2.0,
                                "target_height": 0.1,})
        undesired_contacts = (-1.0, {"threshold": 1})

    rewards = BaseEnvCfg.Rewards(
        only_positive_rewards=False,
        scales = RewardsScales(),
    )

    domain_rand = BaseEnvCfg.DomainRand(
        randomize_friction = True,
        friction_range = [0.1, 1.25],
        randomize_base_mass = True,
        added_mass_range = [-1., 3.],
        push_robots = True,
        push_interval = int(5/0.02),
        max_push_vel_xy = 0.5,
        randomize_initial_state = True
    )

    commands = BaseEnvCfg.Commands(
        heading_command = False,
        ranges = BaseEnvCfg.Commands.Ranges(
            lin_vel_x=(-0.1, 0.1),
            lin_vel_y=(-0.1, 0.1),
            ang_vel_yaw=(-0.1, 0.1)
        ),
        limit_ranges = BaseEnvCfg.Commands.Ranges(
            lin_vel_x=(-0.5, 1.0),
            lin_vel_y=(-0.3, 0.3),
            ang_vel_yaw=(-0.2, 0.2)
        )
    )

    curriculum = BaseEnvCfg.Curriculum(
        enabled = True,
        funcs = {"lin_vel_cmd_levels": lin_vel_cmd_levels}
        )

    default_joint_positions = {
        "g1_dof29": {
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
        }
    }

    callbacks: CallbacksCfg | dict | None = CallbacksCfg(
        startup={
            "material_randomizer": MaterialRandomizer(
                obj_name="g1_dof29",
                static_friction_range = (0.3, 1.0),
                dynamic_friction_range = (0.3, 1.0),
                restitution_range = (0.0, 0.0),
                num_buckets = 64)
        },
        step={"contact_forces": ContactForces(history_length=3)})

@configclass
class WalkG1Dof29EnvRslRlTrainCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 100
    experiment_name = ""  # same as task name
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

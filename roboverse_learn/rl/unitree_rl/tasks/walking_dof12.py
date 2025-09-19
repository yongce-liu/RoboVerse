import torch

from metasim.types import TensorState
from metasim.utils.math import quat_apply, quat_rotate_inverse, wrap_to_pi
from metasim.utils import configclass
from roboverse_learn.rl.unitree_rl.configs.cfg_base import BaseEnvCfg
from roboverse_learn.rl.unitree_rl.envs.env_humanoid import HumanoidEnv
from roboverse_learn.rl.unitree_rl.third_party.isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

'''
@configclass
class WalkingDof12RslRlTrainCfg(RslRlTrainCfg):
    """Environment configuration for 12-DOF walking task."""
    policy = RslRlTrainCfg.Policy(
        init_noise_std = 0.8,
        actor_hidden_dims = [32],
        critic_hidden_dims = [32],
        activation = 'elu',
        rnn_type = 'lstm',
        rnn_hidden_size = 64,
        rnn_num_layers = 1,
    )
    algorithm = RslRlTrainCfg.Algorithm(entropy_coef = 0.01)
    policy_class_name = "ActorCriticRecurrent"
'''

@configclass
class WalkingDof12RslRlTrainCfg(RslRlOnPolicyRunnerCfg):
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


@configclass
class WalkingDof12EnvCfg(BaseEnvCfg):
    num_obs_single = 6 + 3 + 3 * 12 + 2
    obs_len_history = 1
    num_priv_obs_single = 9 + 3 + 3 * 12 + 2
    priv_obs_len_history = 1

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

class WalkingDof12Env(HumanoidEnv):
    def _init_buffers(self):
        self.noise_scale_vec = self._get_noise_scale_vec()
        return super()._init_buffers()

    def _get_noise_scale_vec(self):
        noise_vec = torch.zeros(size=(self.cfg.num_obs_single,), dtype=torch.float, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.cfg.normalization.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = 0.0  # commands
        noise_vec[9 : 9 + self.num_actions] = (
            noise_scales.dof_pos * noise_level * self.cfg.normalization.obs_scales.dof_pos
        )
        noise_vec[9 + self.num_actions : 9 + 2 * self.num_actions] = (
            noise_scales.dof_vel * noise_level * self.cfg.normalization.obs_scales.dof_vel
        )
        noise_vec[9 + 2 * self.num_actions : 9 + 3 * self.num_actions] = 0.0  # previous actions
        noise_vec[9 + 3 * self.num_actions : 9 + 3 * self.num_actions + 2] = 0.0  # sin/cos phase

        return noise_vec

    def _observation(self, env_states: TensorState):
        robot_state = env_states.robots[self.name]
        base_quat = robot_state.root_state[:, 3:7]
        base_lin_vel = quat_rotate_inverse(base_quat, robot_state.root_state[:, 7:10])
        base_ang_vel = quat_rotate_inverse(base_quat, robot_state.root_state[:, 10:13])
        projected_gravity = quat_rotate_inverse(base_quat, self.gravity_vec)

        phase = self.get_leg_phase()
        sin_phase = torch.sin(2 * torch.pi * phase).unsqueeze(1)
        cos_phase = torch.cos(2 * torch.pi * phase).unsqueeze(1)

        q = (env_states.robots[self.name].joint_pos - self.default_dof_pos) * self.cfg.normalization.obs_scales.dof_pos
        dq = env_states.robots[self.name].joint_vel * self.cfg.normalization.obs_scales.dof_vel

        obs_buf = torch.cat((
            base_ang_vel * self.cfg.normalization.obs_scales.ang_vel,  # 3
            projected_gravity,  # 3
            self.commands[:, :3] * self.commands_scale,  # 3
            q,  # num_actions
            dq,  # num_actions
            self.actions,  # num_actions
            sin_phase,  # 1
            cos_phase,  # 1
        ), dim=-1)

        # add noise if needed
        if self.add_noise:
            obs_buf += (2 * torch.rand_like(obs_buf) - 1) * self.noise_scale_vec

        priv_obs_buf = torch.cat((
            base_lin_vel * self.cfg.normalization.obs_scales.lin_vel,
            base_ang_vel * self.cfg.normalization.obs_scales.ang_vel,
            projected_gravity,
            self.commands[:, :3] * self.commands_scale,
            q,  # num_actions
            dq,  # num_actions
            self.actions,
            sin_phase,
            cos_phase,
        ), dim=-1)

        return obs_buf, priv_obs_buf

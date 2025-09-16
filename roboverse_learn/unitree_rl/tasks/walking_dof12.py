import torch
from metasim.utils import configclass
from roboverse_learn.unitree_rl.configs.cfg_base import BaseEnvCfg, RslRlTrainCfg
from roboverse_learn.unitree_rl.envs.env_humanoid import HumanoidEnv

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
        push_interval_s = 5,
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
        # torques = -0.00001
    @configclass
    class RewardExtras(BaseEnvCfg.Rewards.Extras):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.78
    rewards = BaseEnvCfg.Rewards(
        scales = RewardsScales(),
        extras = RewardExtras()
    )

    class InitialStates:
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
    initial_states = InitialStates()

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
    runner = RslRlTrainCfg.Runner(policy_class_name = "ActorCriticRecurrent",)


class WalkingDof12Env(HumanoidEnv):
    def _init_buffers(self):
        self.noise_scale_vec = self._get_noise_scale_vec()
        return super()._init_buffers()

    def _get_noise_scale_vec(self):
        """Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
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

    def _observation(self, env_states):
        phase = self._get_leg_phase()
        sin_phase = torch.sin(2 * torch.pi * phase).unsqueeze(1)
        cos_phase = torch.cos(2 * torch.pi * phase).unsqueeze(1)

        q = (env_states.robots[self.name].joint_pos - self.default_dof_pos) * self.cfg.normalization.obs_scales.dof_pos
        dq = env_states.robots[self.name].joint_vel * self.cfg.normalization.obs_scales.dof_vel

        obs_buf = torch.cat((
            self.base_ang_vel * self.cfg.normalization.obs_scales.ang_vel,  # 3
            self.projected_gravity,  # 3
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
            self.base_lin_vel * self.cfg.normalization.obs_scales.lin_vel,
            self.base_ang_vel * self.cfg.normalization.obs_scales.ang_vel,
            self.projected_gravity,
            self.commands[:, :3] * self.commands_scale,
            q,  # num_actions
            dq,  # num_actions
            self.actions,
            sin_phase,
            cos_phase,
        ), dim=-1)

        return obs_buf, priv_obs_buf

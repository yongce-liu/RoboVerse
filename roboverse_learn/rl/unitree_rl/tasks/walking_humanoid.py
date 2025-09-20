

from typing import Callable

import torch
from metasim.utils import configclass
from roboverse_learn.rl.unitree_rl.configs.cfg_base import BaseEnvCfg, RslRlTrainCfg
from roboverse_learn.rl.unitree_rl.envs.env_humanoid import HumanoidEnv


@configclass
class WalkingHumanoidEnvCfg(BaseEnvCfg):
    """
    Environment configuration for humanoid walking task.
    """
    obs_len_history = 5
    priv_obs_len_history = 5
    class noise(BaseEnvCfg.noise):
        add_noise = True

    class rewards:
        send_timeouts = True
        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        functions: list[Callable] | str = "roboverse_learn.rl.unitree_rl.configs.cfg_reward_funcs"
        class scales:
            termination = -0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0.
            torques = -0.00001
            dof_vel = -0.
            dof_acc = -2.5e-7
            base_height = -0.
            feet_air_time =  1.0
            collision = -1.
            feet_stumble = -0.0
            action_rate = -0.01
            stand_still = -0.

@configclass
class WalkingHumanoidRslRlTrainCfg(RslRlTrainCfg):
    """Train configuration for humanoid walking task for the rsl-rl lib"""
    policy = RslRlTrainCfg.Policy(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlTrainCfg.Algorithm(
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
    runner = RslRlTrainCfg.Runner(
        policy_class_name="ActorCritic",
        num_steps_per_env = 24,
        max_iterations = 50000,
        save_interval = 100,
        experiment_name = "",  # same as task name
        # empirical_normalization = False
    )

class WalkingHumanoidEnv(HumanoidEnv):
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
        q = (env_states.robots[self.name].joint_pos - self.default_dof_pos) * self.cfg.normalization.obs_scales.dof_pos
        dq = env_states.robots[self.name].joint_vel * self.cfg.normalization.obs_scales.dof_vel

        obs_buf = torch.cat((
            self.base_ang_vel * self.cfg.normalization.obs_scales.ang_vel,  # 3
            self.projected_gravity,  # 3
            self.commands[:, :3] * self.commands_scale,  # 3
            q,  # num_actions
            dq,  # num_actions
            self.actions,  # num_actions
            # self.history_buffer['actions'][-1]  # num_actions
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
        ), dim=-1)

        return obs_buf, priv_obs_buf

# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
#
# This file is based on CleanRL's SAC implementation and has been adapted for RoboVerse.
# Original CleanRL code is licensed under MIT License.
from __future__ import annotations

import csv
import os
import random
import time
from typing import Any, Literal

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import gymnasium as gym
import numpy as np
import rootutils
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# RoboVerse imports

rootutils.setup_root(__file__, pythonpath=True)
from gymnasium import make_vec
import metasim  # noqa: F401

from roboverse_learn.rl.clean_rl.buffer import ReplayBuffer
from roboverse_learn.rl.episode_tracker import EpisodeTracker
from roboverse_learn.rl.configs.clean_rl.sac import CleanRLSACConfig
from roboverse_learn.rl.logging.metrics_logger import save_metrics_history


def make_roboverse_env(args):
    """Create RoboVerse environment using make_vec."""
    env_id = f"RoboVerse/{args.task}"
    env = make_vec(
        env_id,
        robots=[args.robot],
        simulator=args.sim,
        num_envs=args.num_envs,
        headless=args.headless,
        cameras=[],
        device=args.device,
    )
    return env


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env, hidden_dim=256):
        super().__init__()
        obs_dim = np.array(env.single_observation_space.shape).prod()
        action_dim = np.prod(env.single_action_space.shape)

        self.fc1 = nn.Linear(obs_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env, hidden_dim=256):
        super().__init__()
        obs_dim = np.array(env.single_observation_space.shape).prod()
        action_dim = np.prod(env.single_action_space.shape)

        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        self.fc_logstd = nn.Linear(hidden_dim, action_dim)
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.single_action_space.high - env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.single_action_space.high + env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


if __name__ == "__main__":

    args = tyro.cli(CleanRLSACConfig)
    args.total_timesteps = args.num_iterations * args.num_envs
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup - use RoboVerse environment
    envs = make_roboverse_env(args)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])

    actor = Actor(envs, hidden_dim=args.actor_hidden_dim).to(device)
    qf1 = SoftQNetwork(envs, hidden_dim=args.critic_hidden_dim).to(device)
    qf2 = SoftQNetwork(envs, hidden_dim=args.critic_hidden_dim).to(device)
    qf1_target = SoftQNetwork(envs, hidden_dim=args.critic_hidden_dim).to(device)
    qf2_target = SoftQNetwork(envs, hidden_dim=args.critic_hidden_dim).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    obs = obs.to(device)
    global_step = 0

    # Track number of gradient updates
    update_step = 0

    # Initialize episode tracker
    episode_tracker = EpisodeTracker(args.num_envs, device)

    # Initialize metrics tracking
    model_dir = f"runs/{run_name}"
    metrics_path = os.path.join(model_dir, "metrics.csv")
    metrics_history: list[dict[str, Any]] = []

    # Create progress bar for total iterations (each iteration collects num_envs steps)
    pbar = tqdm(total=args.num_iterations, desc="SAC Training")

    for iteration in range(1, args.num_iterations + 1):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = torch.tensor([envs.single_action_space.sample() for _ in range(envs.num_envs)], device=device)
        else:
            actions, _, _ = actor.get_action(obs)
            actions = actions.detach()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        next_obs = next_obs.to(device)


        # Compute 'true' next_obs for saving (similar to fast_td3)
        true_next_obs = torch.where(truncations[:, None] > 0, infos["observations"]["raw"]["obs"], next_obs)
        rb.add(obs.cpu().numpy(), true_next_obs.cpu().numpy(), actions.cpu().numpy(), rewards.cpu().numpy(), terminations.cpu().numpy(), infos)

        # Update episode tracker
        episode_tracker.update(rewards, terminations, truncations)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs
        global_step += args.num_envs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # Compute TD error statistics (for logging)
            with torch.no_grad():
                td_error = (next_q_value - qf1_a_values).detach()
                td_error_mean = td_error.mean()
                td_error_std = td_error.std()
                td_error_abs_mean = td_error.abs().mean()

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                list(qf1.parameters()) + list(qf2.parameters()),
                max_norm=float('inf')
            )
            q_optimizer.step()

            update_step += 1

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                        actor.parameters(),
                        max_norm=float('inf')
                    )
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % args.log_interval == 0:
                # Compute action statistics
                with torch.no_grad():
                    current_actions, _, _ = actor.get_action(data.observations)
                    action_mean = current_actions.mean(dim=0).cpu().numpy()
                    action_std = current_actions.std(dim=0).cpu().numpy()
                    action_l2_norm = torch.norm(current_actions, p=2, dim=-1).mean().cpu().item()

                # Compute buffer usage
                buffer_usage = rb.pos / args.buffer_size

                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                writer.add_scalar("losses/critic_grad_norm", critic_grad_norm.item(), global_step)
                writer.add_scalar("losses/actor_grad_norm", actor_grad_norm.item(), global_step)
                writer.add_scalar("losses/td_error_mean", td_error_mean.item(), global_step)
                writer.add_scalar("losses/td_error_std", td_error_std.item(), global_step)
                writer.add_scalar("actions/action_mean", np.mean(action_mean), global_step)
                writer.add_scalar("actions/action_std", np.mean(action_std), global_step)
                writer.add_scalar("actions/action_l2_norm", action_l2_norm, global_step)
                writer.add_scalar("buffer/usage", buffer_usage, global_step)

                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

                # Log episode statistics
                episode_stats = episode_tracker.get_detailed_stats()
                if episode_tracker.get_episode_count() > 0:
                    writer.add_scalar("charts/episodic_return_mean", episode_stats['return_mean'], global_step)
                    writer.add_scalar("charts/episodic_return_std", episode_stats['return_std'], global_step)
                    writer.add_scalar("charts/episodic_length_mean", episode_stats['length_mean'], global_step)
                    writer.add_scalar("charts/episodic_length_std", episode_stats['length_std'], global_step)

                writer.add_scalar("charts/updates", update_step, global_step)
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                # Compute wall clock time and SPS for metrics
                wall_clock_time = time.time() - start_time
                sps = int(global_step / wall_clock_time)

                # Build metrics entry for CSV export
                metrics_entry = {
                    "global_step": int(global_step),
                    "iteration": int(iteration),
                    "updates": int(update_step),
                    "speed": float(sps),
                    "wall_clock_time": float(wall_clock_time),
                    "qf1_values": float(qf1_a_values.mean().item()),
                    "qf2_values": float(qf2_a_values.mean().item()),
                    "qf1_loss": float(qf1_loss.item()),
                    "qf2_loss": float(qf2_loss.item()),
                    "qf_loss": float(qf_loss.item() / 2.0),
                    "actor_loss": float(actor_loss.item()),
                    "alpha": float(alpha),
                    "critic_grad_norm": float(critic_grad_norm.item()),
                    "actor_grad_norm": float(actor_grad_norm.item()),
                    "td_error_mean": float(td_error_mean.item()),
                    "td_error_std": float(td_error_std.item()),
                    "action_mean": float(np.mean(action_mean)),
                    "action_std": float(np.mean(action_std)),
                    "action_l2_norm": float(action_l2_norm),
                    "buffer_usage": float(buffer_usage),
                }
                if args.autotune:
                    metrics_entry["alpha_loss"] = float(alpha_loss.item())
                if episode_tracker.get_episode_count() > 0:
                    metrics_entry.update({
                        "episodic_return_mean": float(episode_stats['return_mean']),
                        "episodic_return_std": float(episode_stats['return_std']),
                        "episodic_length_mean": float(episode_stats['length_mean']),
                        "episodic_length_std": float(episode_stats['length_std']),
                        "episode_count": int(episode_tracker.get_episode_count()),
                    })
                metrics_history.append(metrics_entry)

                # Save checkpoint every save_interval steps
                if args.save_interval > 0 and global_step % (args.save_interval * args.log_interval) == 0:
                    checkpoint_path = os.path.join(model_dir, f"checkpoint_{global_step}.pt")
                    os.makedirs(model_dir, exist_ok=True)
                    torch.save({
                        'global_step': global_step,
                        'actor_state_dict': actor.state_dict(),
                        'qf1_state_dict': qf1.state_dict(),
                        'qf2_state_dict': qf2.state_dict(),
                        'actor_optimizer_state_dict': actor_optimizer.state_dict(),
                        'q_optimizer_state_dict': q_optimizer.state_dict(),
                    }, checkpoint_path)
                    print(f"Saved checkpoint to {checkpoint_path}")
                    # Also save metrics
                    save_metrics_history(metrics_path, metrics_history)

                if episode_tracker.get_episode_count() > 0:
                    pbar.set_postfix({
                        'return': f"{episode_stats['return_mean']:.2f}",
                        'length': f"{episode_stats['length_mean']:.1f}",
                        'SPS': sps
                    })

        # Update progress bar once per iteration
        pbar.update(1)

    # Save final metrics
    pbar.close()
    save_metrics_history(metrics_path, metrics_history)

    envs.close()
    writer.close()

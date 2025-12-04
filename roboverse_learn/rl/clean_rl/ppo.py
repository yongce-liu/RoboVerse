# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
#
# This file is based on CleanRL's PPO implementation and has been adapted for RoboVerse.
# Original CleanRL code is licensed under MIT License.
from __future__ import annotations

import csv
import os
import random
import time
from typing import Any

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import gymnasium as gym
import numpy as np
import rootutils
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# RoboVerse imports

rootutils.setup_root(__file__, pythonpath=True)
from gymnasium import make_vec
import metasim  # noqa: F401

from roboverse_learn.rl.episode_tracker import EpisodeTracker
from roboverse_learn.rl.configs.clean_rl.ppo import CleanRLPPOConfig
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


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer




class Agent(nn.Module):
    def __init__(self, envs, actor_hidden_dim=64, critic_hidden_dim=64):
        super().__init__()
        obs_dim = np.array(envs.single_observation_space.shape).prod()
        action_dim = np.prod(envs.single_action_space.shape)

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, critic_hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(critic_hidden_dim, critic_hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(critic_hidden_dim, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, actor_hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(actor_hidden_dim, actor_hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(actor_hidden_dim, action_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


if __name__ == "__main__":
    args = tyro.cli(CleanRLPPOConfig)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
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

    agent = Agent(envs, actor_hidden_dim=args.actor_hidden_dim, critic_hidden_dim=args.critic_hidden_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs), dtype=torch.bool, device=device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    update_step = 0  # Track number of gradient updates
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = next_obs.to(device)
    next_done = torch.zeros(args.num_envs, dtype=torch.bool, device=device)

    # Initialize episode tracker
    episode_tracker = EpisodeTracker(args.num_envs, device)

    # Initialize metrics tracking
    model_dir = f"runs/{run_name}"
    metrics_path = os.path.join(model_dir, "metrics.csv")
    metrics_history: list[dict[str, Any]] = []

    start_time = time.time()
    pbar = tqdm(total=args.num_iterations, desc="PPO Training")
    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action)
            next_done = torch.logical_or(terminations, truncations)
            rewards[step] = reward.view(-1)
            dones[step] = next_done

            # Update episode tracker
            episode_tracker.update(reward.view(-1), terminations, truncations)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = (~next_done).float()
                    nextvalues = next_value
                else:
                    nextnonterminal = (~dones[t + 1]).float()
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        grad_norms = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                grad_norms.append(grad_norm.item())
                optimizer.step()

                update_step += 1

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Compute TD error statistics
        td_error = b_returns.cpu().numpy() - b_values.cpu().numpy()
        td_error_mean = np.mean(td_error)
        td_error_std = np.std(td_error)
        td_error_abs_mean = np.mean(np.abs(td_error))

        # Compute action statistics
        action_mean = actions.mean(dim=(0, 1)).cpu().numpy()
        action_std = actions.std(dim=(0, 1)).cpu().numpy()

        # Compute reward statistics (per-step rewards)
        reward_mean = rewards.mean().cpu().item()
        reward_std = rewards.std().cpu().item()

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("losses/grad_norm", np.mean(grad_norms), global_step)
        writer.add_scalar("losses/td_error_mean", td_error_mean, global_step)
        writer.add_scalar("losses/td_error_std", td_error_std, global_step)
        writer.add_scalar("actions/action_mean", np.mean(action_mean), global_step)
        writer.add_scalar("actions/action_std", np.mean(action_std), global_step)
        writer.add_scalar("rewards/reward_mean", reward_mean, global_step)
        writer.add_scalar("rewards/reward_std", reward_std, global_step)

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
        # Note: We use `iteration` for the CSV file's global_step for consistency with previous logging,
        # and store the true environment step count in the `frame` field.
        metrics_entry = {
            "global_step": int(iteration),
            "iteration": int(iteration),
            "updates": int(update_step),
            "speed": float(sps),
            "frame": int(global_step),
            "wall_clock_time": float(wall_clock_time),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "value_loss": float(v_loss.item()),
            "policy_loss": float(pg_loss.item()),
            "entropy": float(entropy_loss.item()),
            "old_approx_kl": float(old_approx_kl.item()),
            "approx_kl": float(approx_kl.item()),
            "clipfrac": float(np.mean(clipfracs)),
            "explained_variance": float(explained_var),
            "grad_norm": float(np.mean(grad_norms)),
            "td_error_mean": float(td_error_mean),
            "td_error_std": float(td_error_std),
            "action_mean": float(np.mean(action_mean)),
            "action_std": float(np.mean(action_std)),
            "reward_mean": float(reward_mean),
            "reward_std": float(reward_std),
        }
        if episode_tracker.get_episode_count() > 0:
            metrics_entry.update({
                "episodic_return_mean": float(episode_stats['return_mean']),
                "episodic_return_std": float(episode_stats['return_std']),
                "episodic_length_mean": float(episode_stats['length_mean']),
                "episodic_length_std": float(episode_stats['length_std']),
                "episode_count": int(episode_tracker.get_episode_count()),
            })
        metrics_history.append(metrics_entry)

        # Save checkpoint every save_interval iterations
        if args.save_interval > 0 and iteration % args.save_interval == 0:
            checkpoint_path = os.path.join(model_dir, f"checkpoint_{iteration}.pt")
            os.makedirs(model_dir, exist_ok=True)
            torch.save({
                'iteration': iteration,
                'model_state_dict': agent.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
            # Also save metrics
            save_metrics_history(metrics_path, metrics_history)

        # Update progress bar
        pbar.update(1)
        if episode_tracker.get_episode_count() > 0:
            pbar.set_postfix({
                'return': f"{episode_stats['return_mean']:.2f}",
                'length': f"{episode_stats['length_mean']:.1f}",
                'SPS': sps
            })

    # Save final metrics
    pbar.close()
    save_metrics_history(metrics_path, metrics_history)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

        # Save model for RoboVerse inference
        roboverse_model_path = f"get_started/output/rl/clean_rl_{args.task}_{args.sim}"
        torch.save(agent.state_dict(), roboverse_model_path)
        print(f"RoboVerse model saved to {roboverse_model_path}")

    envs.close()
    writer.close()

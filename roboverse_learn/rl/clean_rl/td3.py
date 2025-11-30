# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/td3/#td3_continuous_actionpy
#
# This file is based on CleanRL's TD3 implementation and has been adapted for RoboVerse.
# Original CleanRL code is licensed under MIT License.
from __future__ import annotations

import csv
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Literal

# RoboVerse imports
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


rootutils.setup_root(__file__, pythonpath=True)
from gymnasium import make_vec
import metasim  # noqa: F401

from roboverse_learn.rl.clean_rl.buffer import ReplayBuffer
from roboverse_learn.rl.episode_tracker import EpisodeTracker


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""
    save_interval: int = 25
    """save checkpoint every N iterations"""

    # RoboVerse specific arguments
    task: str = "reach_origin"
    """the RoboVerse task name"""
    robot: str = "franka"
    """the robot type"""
    sim: Literal["isaaclab", "isaacgym", "mujoco", "genesis", "mjx"] = "mjx"
    """the simulator backend"""
    headless: bool = False
    """whether to run in headless mode"""
    device: str = "cuda"
    """device to run on"""

    # Algorithm specific arguments
    """the id of the environment"""
    total_timesteps: int = 100000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 4096
    """the number of parallel game environments"""
    buffer_size: int = int(1e7)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 4096
    """the batch size of sample from the reply memory"""
    policy_noise: float = 0.2
    """the scale of policy noise"""
    exploration_noise: float = 0.05
    """the scale of exploration noise"""
    learning_starts: int = 25000
    """timestep to start learning"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""
    log_interval: int = 262144
    """interval (in samples) for logging metrics to match PPO's iteration"""



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


def save_metrics_history(save_path: str, metrics_history: list[dict[str, Any]]) -> None:
    """Write aggregated metrics to a CSV file for readability."""
    if not metrics_history:
        return

    fieldnames: list[str] = []
    for entry in metrics_history:
        for key in entry.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_history)
    print(f"Saved metrics history to {save_path}")


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape),
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, np.prod(env.single_action_space.shape))
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
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias


if __name__ == "__main__":

    args = tyro.cli(Args)
    model_dir = os.path.join("models", args.exp_name, args.task)
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=args.exp_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(model_dir)
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

    actor = Actor(envs).to(device)
    qf1 = QNetwork(envs).to(device)
    qf2 = QNetwork(envs).to(device)
    qf1_target = QNetwork(envs).to(device)
    qf2_target = QNetwork(envs).to(device)
    target_actor = Actor(envs).to(device)
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.learning_rate)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.learning_rate)

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
    update_step = 0  # Track number of gradient updates

    # Initialize episode tracker
    episode_tracker = EpisodeTracker(args.num_envs, device)

    # Initialize metrics tracking
    metrics_path = os.path.join(model_dir, "metrics.csv")
    metrics_history: list[dict[str, Any]] = []

    # Progress bar and logging iteration counter.
    # We keep `global_step` as the true environment step count (frames),
    # and use `iteration` as the logging/progress-step index
    # to mirror the style of PPO (logging every log_interval samples).
    total_iterations = (args.total_timesteps + args.log_interval - 1) // args.log_interval
    iteration = 0
    last_log_step = 0

    # Track rewards over each iteration for proper averaging
    iteration_rewards = []

    pbar = tqdm(total=total_iterations, desc="TD3 Training")
    while global_step < args.total_timesteps:
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
             actions = torch.tensor([envs.single_action_space.sample() for _ in range(envs.num_envs)], device=device)
        else:
            with torch.no_grad():
                actions = actor(torch.Tensor(obs).to(device))
                actions += torch.normal(0, actor.action_scale * args.exploration_noise)
                # Clamp actions to valid action space bounds
                actions = actions.clamp(
                    envs.single_action_space.low[0],
                    envs.single_action_space.high[0]
                )
        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        next_obs = next_obs.to(device)
        dones = torch.logical_or(terminations, truncations)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        # Compute 'true' next_obs for saving (similar to fast_td3)
        true_next_obs = torch.where(truncations[:, None] > 0, infos["observations"]["raw"]["obs"], next_obs)
        rb.add(obs.cpu().numpy(), true_next_obs.cpu().numpy(), actions.cpu().numpy(), rewards.cpu().numpy(), terminations.cpu().numpy(), infos)

        # Update episode tracker
        episode_tracker.update(rewards, dones)

        # Accumulate rewards for iteration averaging
        iteration_rewards.append(rewards.mean().cpu().item())

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs
        global_step += args.num_envs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                clipped_noise = (torch.randn_like(data.actions, device=device) * args.policy_noise).clamp(
                    -args.noise_clip, args.noise_clip
                ) * target_actor.action_scale

                next_state_actions = (target_actor(data.next_observations) + clipped_noise).clamp(
                    envs.single_action_space.low[0], envs.single_action_space.high[0]
                )
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # Compute TD error statistics
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

            if global_step % args.policy_frequency == 0:
                actor_loss = -qf1(data.observations, actor(data.observations)).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                    actor.parameters(),
                    max_norm=float('inf')
                )
                actor_optimizer.step()

                # update the target network
                for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            # Compute action statistics
            with torch.no_grad():
                current_actions = actor(obs)
                action_mean = current_actions.mean(dim=0).cpu().numpy()
                action_std = current_actions.std(dim=0).cpu().numpy()
                action_l2_norm = torch.norm(current_actions, p=2, dim=-1).mean().cpu().item()

            # Compute buffer usage
            buffer_usage = rb.pos / args.buffer_size

            # Compute reward statistics (averaged over iteration, not just current step)
            reward_mean = np.mean(iteration_rewards) if iteration_rewards else 0.0
            reward_std = np.std(iteration_rewards) if iteration_rewards else 0.0

            writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
            writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
            writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
            writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
            writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
            writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
            writer.add_scalar("losses/critic_grad_norm", critic_grad_norm.item(), global_step)
            writer.add_scalar("losses/actor_grad_norm", actor_grad_norm.item(), global_step)
            writer.add_scalar("losses/td_error_mean", td_error_mean.item(), global_step)
            writer.add_scalar("losses/td_error_std", td_error_std.item(), global_step)
            writer.add_scalar("actions/action_mean", np.mean(action_mean), global_step)
            writer.add_scalar("actions/action_std", np.mean(action_std), global_step)
            writer.add_scalar("rewards/reward_mean", reward_mean, global_step)
            writer.add_scalar("rewards/reward_std", reward_std, global_step)
            writer.add_scalar("buffer/usage", buffer_usage, global_step)

            # Log episode statistics and metrics at fixed intervals (matching PPO's iteration)
            if global_step - last_log_step >= args.log_interval:
                iteration += 1
                last_log_step = global_step
                pbar.update(1)

                episode_stats = episode_tracker.get_detailed_stats()
                sps = int(global_step / (time.time() - start_time))
                wall_clock_time = time.time() - start_time
                if episode_tracker.get_episode_count() > 0:
                    writer.add_scalar("charts/episodic_return_mean", episode_stats['return_mean'], global_step)
                    writer.add_scalar("charts/episodic_return_std", episode_stats['return_std'], global_step)
                    writer.add_scalar("charts/episodic_length_mean", episode_stats['length_mean'], global_step)
                    writer.add_scalar("charts/episodic_length_std", episode_stats['length_std'], global_step)
                    print(f"SPS: {sps}, return: {episode_stats['return_mean']:.2f}±{episode_stats['return_std']:.2f}, length: {episode_stats['length_mean']:.1f}, timesteps: {global_step}")
                else:
                    print(f"SPS: {sps}, timesteps: {global_step}")
                writer.add_scalar("charts/SPS", sps, global_step)
                writer.add_scalar("charts/updates", update_step, global_step)
                writer.add_scalar("charts/wall_clock_time", wall_clock_time, global_step)

                # Accumulate metrics for CSV logging
                # Use iteration index as the logging step (matching PPO),
                # and store the true environment step count in the `frame` field.
                metrics_entry = {
                    "global_step": int(iteration),
                    "iteration": int(iteration),
                    "updates": int(update_step),
                    "speed": float(sps),
                    "frame": int(global_step),
                    "wall_clock_time": float(wall_clock_time),
                    "qf1_values": float(qf1_a_values.mean().item()),
                    "qf2_values": float(qf2_a_values.mean().item()),
                    "qf1_loss": float(qf1_loss.item()),
                    "qf2_loss": float(qf2_loss.item()),
                    "qf_loss": float(qf_loss.item() / 2.0),
                    "actor_loss": float(actor_loss.item()),
                    "critic_grad_norm": float(critic_grad_norm.item()),
                    "actor_grad_norm": float(actor_grad_norm.item()),
                    "td_error_mean": float(td_error_mean.item()),
                    "td_error_std": float(td_error_std.item()),
                    "action_mean": float(np.mean(action_mean)),
                    "action_std": float(np.mean(action_std)),
                    "reward_mean": float(reward_mean),
                    "reward_std": float(reward_std),
                    "buffer_usage": float(buffer_usage),
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

                # Reset iteration rewards for next logging period
                iteration_rewards = []

                # Save checkpoint every save_interval iterations
                if args.save_interval > 0 and iteration % args.save_interval == 0:
                    checkpoint_path = os.path.join(model_dir, f"checkpoint_iter_{iteration}.pt")
                    torch.save((actor.state_dict(), qf1.state_dict(), qf2.state_dict()), checkpoint_path)
                    print(f"Checkpoint saved at iteration {iteration}: {checkpoint_path}")
                    # Also save metrics at checkpoint
                    save_metrics_history(metrics_path, metrics_history)

    pbar.close()
    # Save metrics history
    save_metrics_history(metrics_path, metrics_history)

    if args.save_model:
        model_path = os.path.join(model_dir, f"{args.exp_name}.cleanrl_model")
        torch.save((actor.state_dict(), qf1.state_dict(), qf2.state_dict()), model_path)
        print(f"model saved to {model_path}")
    envs.close()
    writer.close()

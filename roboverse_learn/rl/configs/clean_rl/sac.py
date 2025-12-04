from typing import Optional

from metasim.utils import configclass

from roboverse_learn.rl.configs.clean_rl.base import BaseRLConfig, SimBackend


@configclass
class CleanRLSACConfig(BaseRLConfig):
    """CleanRL SAC configuration adapted for RoboVerse."""

    # Experiment
    exp_name: str = "sac"

    # Tracking / logging flags (CleanRL-style)
    track: bool = False
    wandb_project: str = "cleanRL"
    wandb_entity: Optional[str] = None
    capture_video: bool = False
    save_model: bool = False
    log_interval: int = 262144
    save_interval: int = 25

    # RoboVerse specific arguments
    task: str = "reach_origin"
    robot: str = "franka"
    sim: SimBackend = "mjx"
    headless: bool = False
    device: str = "cuda"

    # Algorithm specific arguments
    total_timesteps: int = 100000000
    num_envs: int = 4096
    buffer_size: int = int(1e7)
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 4096
    learning_starts: int = 25000
    policy_lr: float = 3e-4
    q_lr: float = 1e-3
    policy_frequency: int = 2
    target_network_frequency: int = 1
    alpha: float = 0.2
    autotune: bool = True

    # Network architecture
    actor_hidden_dim: int = 256
    critic_hidden_dim: int = 256



__all__ = ["CleanRLSACConfig"]

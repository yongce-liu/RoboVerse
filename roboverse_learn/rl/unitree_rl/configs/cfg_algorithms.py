from __future__ import annotations

"""Base class for algorithms."""

from metasim.utils import configclass

@configclass
class Algorithm:
    """Training config class for PPO."""

    value_loss_coef = 1.0
    """Value loss coefficient."""
    use_clipped_value_loss = True
    """Use clipped value loss."""
    clip_param = 0.2
    """Clipping parameter for PPO."""
    entropy_coef = 0.01
    """Entropy coefficient."""
    num_learning_epochs = 5
    """Number of learning epochs."""
    num_mini_batches = 4
    """mini batch size = num_envs*n_steps / num_mini_batches"""
    learning_rate = 1.0e-3
    schedule = "adaptive"
    gamma = 0.99
    lam = 0.95
    desired_kl = 0.01
    max_grad_norm = 1.0

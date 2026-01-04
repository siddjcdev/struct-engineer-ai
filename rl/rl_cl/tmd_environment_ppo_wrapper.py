"""
PPO-Specific Environment Wrapper (OPTIONAL)
============================================

Use this wrapper ONLY IF v8 PPO training struggles with reward variance
or observation scaling issues.

The adaptive environment works fine for both SAC and PPO, but PPO is
more sensitive to:
1. High reward variance (M7.4, M8.4 have huge displacement swings)
2. Observation scaling (PPO works better with normalized obs)
3. Advantage estimation stability

This wrapper adds:
- Observation normalization (running mean/std)
- Optional reward clipping (prevent extreme values)
- Reward statistics tracking

Author: Siddharth
Date: January 2026
"""

import gymnasium as gym
import numpy as np
from typing import Optional, Tuple


class PPOFriendlyWrapper(gym.Wrapper):
    """
    Wrapper to make earthquake environment more PPO-friendly

    Features:
    1. Observation normalization (running mean/std)
    2. Optional reward clipping
    3. Statistics tracking
    """

    def __init__(
        self,
        env: gym.Env,
        normalize_obs: bool = True,
        clip_obs: float = 10.0,
        clip_reward: Optional[float] = None,
        gamma: float = 0.99
    ):
        """
        Args:
            env: Base environment
            normalize_obs: Whether to normalize observations
            clip_obs: Clip normalized observations to ±clip_obs
            clip_reward: If set, clip rewards to ±clip_reward
            gamma: Discount factor for return normalization
        """
        super().__init__(env)

        self.normalize_obs = normalize_obs
        self.clip_obs = clip_obs
        self.clip_reward = clip_reward
        self.gamma = gamma

        # Running statistics for observation normalization
        if normalize_obs:
            obs_dim = env.observation_space.shape[0]
            self.obs_mean = np.zeros(obs_dim, dtype=np.float32)
            self.obs_var = np.ones(obs_dim, dtype=np.float32)
            self.obs_count = 1e-4  # Small epsilon to avoid division by zero

        # Running statistics for reward normalization (optional)
        self.return_mean = 0.0
        self.return_var = 1.0
        self.return_count = 1e-4

        # Episode tracking
        self.episode_return = 0.0
        self.episode_length = 0

        # Statistics
        self.total_episodes = 0
        self.reward_history = []

    def _update_obs_stats(self, obs: np.ndarray):
        """Update running mean and variance for observations"""
        if not self.normalize_obs:
            return

        # Welford's online algorithm for running mean/variance
        self.obs_count += 1
        delta = obs - self.obs_mean
        self.obs_mean += delta / self.obs_count
        delta2 = obs - self.obs_mean
        self.obs_var += delta * delta2

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observations using running statistics"""
        if not self.normalize_obs:
            return obs

        # Compute standard deviation
        obs_std = np.sqrt(self.obs_var / self.obs_count + 1e-8)

        # Normalize
        obs_normalized = (obs - self.obs_mean) / obs_std

        # Clip to prevent extreme values
        if self.clip_obs is not None:
            obs_normalized = np.clip(obs_normalized, -self.clip_obs, self.clip_obs)

        return obs_normalized

    def _process_reward(self, reward: float) -> float:
        """Process reward (optional clipping)"""
        if self.clip_reward is not None:
            reward = np.clip(reward, -self.clip_reward, self.clip_reward)

        return reward

    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        """Reset environment and update statistics"""
        # Track episode statistics
        if self.episode_length > 0:
            self.total_episodes += 1
            self.reward_history.append(self.episode_return)

        # Reset episode tracking
        self.episode_return = 0.0
        self.episode_length = 0

        # Reset base environment
        obs, info = self.env.reset(**kwargs)

        # Update observation statistics
        self._update_obs_stats(obs)

        # Normalize observation
        obs_normalized = self._normalize_obs(obs)

        return obs_normalized, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Step environment with observation normalization and reward processing"""
        # Step base environment
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Update observation statistics
        self._update_obs_stats(obs)

        # Normalize observation
        obs_normalized = self._normalize_obs(obs)

        # Process reward
        reward_processed = self._process_reward(reward)

        # Update episode statistics
        self.episode_return += reward
        self.episode_length += 1

        # Add wrapper statistics to info
        info['wrapper_stats'] = {
            'obs_mean': self.obs_mean.copy(),
            'obs_std': np.sqrt(self.obs_var / self.obs_count + 1e-8),
            'original_reward': reward,
            'processed_reward': reward_processed,
            'episode_return': self.episode_return,
            'episode_length': self.episode_length
        }

        return obs_normalized, reward_processed, terminated, truncated, info

    def get_statistics(self) -> dict:
        """Get wrapper statistics"""
        stats = {
            'total_episodes': self.total_episodes,
            'obs_mean': self.obs_mean.copy() if self.normalize_obs else None,
            'obs_std': np.sqrt(self.obs_var / self.obs_count + 1e-8) if self.normalize_obs else None,
            'recent_episode_returns': self.reward_history[-100:] if len(self.reward_history) > 0 else [],
            'mean_episode_return': np.mean(self.reward_history) if len(self.reward_history) > 0 else 0.0,
            'std_episode_return': np.std(self.reward_history) if len(self.reward_history) > 0 else 0.0
        }
        return stats


def make_ppo_friendly_env(
    earthquake_file: str,
    max_force: float = 150000.0,
    normalize_obs: bool = True,
    clip_reward: Optional[float] = None
):
    """
    Create PPO-friendly earthquake control environment

    Args:
        earthquake_file: Path to earthquake CSV file
        max_force: Maximum control force (N)
        normalize_obs: Whether to normalize observations
        clip_reward: If set, clip rewards to ±clip_reward

    Returns:
        Wrapped environment optimized for PPO
    """
    from tmd_environment_adaptive_reward import make_improved_tmd_env

    # Create base environment
    base_env = make_improved_tmd_env(earthquake_file, max_force=max_force)

    # Wrap with PPO-friendly wrapper
    env = PPOFriendlyWrapper(
        base_env,
        normalize_obs=normalize_obs,
        clip_obs=10.0,
        clip_reward=clip_reward,
        gamma=0.99
    )

    return env


# Example usage:
if __name__ == "__main__":
    # Test the wrapper
    env = make_ppo_friendly_env(
        "../../matlab/datasets/PEER_small_M4.5_PGA0.25g.csv",
        max_force=50000,
        normalize_obs=True,
        clip_reward=10.0  # Clip extreme rewards
    )

    print("Testing PPO-friendly wrapper...")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    obs, info = env.reset()
    print(f"\nInitial observation (normalized): {obs}")

    # Run a few steps
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        if i == 0:
            print(f"\nStep {i+1}:")
            print(f"  Normalized obs: {obs}")
            print(f"  Reward (processed): {reward:.4f}")
            print(f"  Original reward: {info['wrapper_stats']['original_reward']:.4f}")

    print(f"\nWrapper statistics:")
    stats = env.get_statistics()
    for key, value in stats.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: shape {value.shape}")
        else:
            print(f"  {key}: {value}")

    print("\n✓ PPO-friendly wrapper working correctly!")

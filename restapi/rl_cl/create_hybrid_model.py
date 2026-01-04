"""
CREATE HYBRID MODEL: Combine Old (Peak Disp) + New (DCR) Models

Strategy: Create an ensemble that combines:
- Old model: Good at peak displacement reduction
- New model: Good at DCR (drift distribution)

Approach: Weighted average of actions based on current state
- Early in earthquake (high displacement): Weight towards old model
- Mid-late earthquake (drift building): Weight towards new model
"""

import numpy as np
import torch
from stable_baselines3 import SAC
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rl_cl.rl_cl_tmd_environment import ImprovedTMDBuildingEnv
import gymnasium as gym
from gymnasium import spaces


class Simple4DWrapper(gym.Wrapper):
    """
    Wrapper to convert 8D observation space to 4D for compatibility with old models

    Extracts only: [roof_disp, roof_vel, tmd_disp, tmd_vel]
    From 8D: [roof_disp, roof_vel, floor8_disp, floor8_vel, floor6_disp, floor6_vel, tmd_disp, tmd_vel]
    """

    def __init__(self, env):
        super().__init__(env)
        # Override observation space to 4D
        self.observation_space = spaces.Box(
            low=np.array([-1.2, -3.0, -1.5, -3.5]),
            high=np.array([1.2, 3.0, 1.5, 3.5]),
            dtype=np.float32
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Extract: roof (indices 0-1) + tmd (indices 6-7)
        obs_4d = np.concatenate([obs[0:2], obs[6:8]])
        return obs_4d, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Extract: roof (indices 0-1) + tmd (indices 6-7)
        obs_4d = np.concatenate([obs[0:2], obs[6:8]])
        return obs_4d, reward, terminated, truncated, info

    def get_episode_metrics(self):
        """Pass through to wrapped environment"""
        return self.env.get_episode_metrics()


def make_simple_4d_env(earthquake_file):
    """Create 4D environment compatible with old models"""
    data = np.loadtxt(earthquake_file, delimiter=',', skiprows=1)

    if data.shape[1] >= 2:
        accelerations = data[:, 1]
        dt = float(np.mean(np.diff(data[:, 0])))
    else:
        accelerations = data
        dt = 0.02

    env = ImprovedTMDBuildingEnv(
        earthquake_data=accelerations,
        dt=dt,
        max_force=150000.0,
        earthquake_name=str(earthquake_file)
    )

    # Wrap to provide 4D observations
    env = Simple4DWrapper(env)
    return env


class HybridRLModel:
    """
    Hybrid model combining two RL models with complementary strengths

    Strategy:
    - Model 1 (Old): Optimized for peak displacement reduction
    - Model 2 (New): Optimized for DCR (drift uniformity)
    - Combine using adaptive weighting based on episode progress and state
    """

    def __init__(self, model1_path, model2_path, strategy='adaptive'):
        """
        Args:
            model1_path: Path to old model (good peak displacement)
            model2_path: Path to new model (good DCR)
            strategy: 'adaptive', 'weighted_avg', or 'max_response'
        """
        print("=" * 70)
        print("CREATING HYBRID MODEL")
        print("=" * 70)

        self.model1 = SAC.load(model1_path)
        self.model2 = SAC.load(model2_path)
        self.strategy = strategy

        print(f"\nModel 1 (Peak Disp): {model1_path}")
        print(f"Model 2 (DCR):       {model2_path}")
        print(f"Strategy:            {strategy}")

        # Track episode progress for adaptive weighting
        self.step_count = 0
        self.max_steps = 2000  # Will be updated

    def reset_episode(self, max_steps=2000):
        """Reset tracking for new episode"""
        self.step_count = 0
        self.max_steps = max_steps

    def predict(self, observation, deterministic=True):
        """
        Predict action by combining both models

        Returns:
            action, None (to match SAC interface)
        """
        self.step_count += 1

        # Get actions from both models
        action1, _ = self.model1.predict(observation, deterministic=deterministic)
        action2, _ = self.model2.predict(observation, deterministic=deterministic)

        if self.strategy == 'adaptive':
            # Adaptive weighting based on episode progress
            # Early: Focus on peak displacement (model1)
            # Late: Focus on DCR uniformity (model2)
            progress = self.step_count / self.max_steps

            # Sigmoid transition from model1 to model2
            # Early (0-30%): 80% model1, 20% model2
            # Mid (30-70%): Gradual transition
            # Late (70-100%): 20% model1, 80% model2
            weight1 = 0.8 * (1 - progress) + 0.2
            weight2 = 1 - weight1

            action = weight1 * action1 + weight2 * action2

        elif self.strategy == 'weighted_avg':
            # Simple 50-50 weighted average
            action = 0.5 * action1 + 0.5 * action2

        elif self.strategy == 'max_response':
            # Use whichever model suggests stronger action
            # (More aggressive control)
            if abs(action1[0]) > abs(action2[0]):
                action = action1
            else:
                action = action2

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        # Clip to valid action range
        action = np.clip(action, -1.0, 1.0)

        return action, None


def test_hybrid_model(model1_path, model2_path, earthquake_file, strategy='adaptive'):
    """
    Test hybrid model on earthquake

    Args:
        model1_path: Path to old model
        model2_path: Path to new model
        earthquake_file: Earthquake to test on
        strategy: Combination strategy

    Returns:
        metrics: Performance metrics
    """
    print("\n" + "=" * 70)
    print(f"TESTING HYBRID MODEL ({strategy.upper()} STRATEGY)")
    print("=" * 70)

    # Create hybrid model
    hybrid = HybridRLModel(model1_path, model2_path, strategy=strategy)

    # Create environment (4D compatible)
    env = make_simple_4d_env(earthquake_file)

    # Reset
    obs, info = env.reset()
    # Access the wrapped environment's earthquake data
    hybrid.reset_episode(max_steps=len(env.unwrapped.earthquake_data))

    # Run episode
    done = False
    step = 0

    while not done:
        action, _ = hybrid.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1

        if step % 500 == 0:
            print(f"  Step {step}/{len(env.unwrapped.earthquake_data)}: "
                  f"Roof disp = {obs[0]*100:.2f} cm, "
                  f"Reward = {reward:.2f}")

    # Get metrics
    metrics = env.get_episode_metrics()

    print(f"\n{strategy.upper()} STRATEGY RESULTS:")
    print(f"  Peak displacement: {metrics['peak_roof_displacement']*100:.2f} cm")
    print(f"  RMS displacement:  {metrics['rms_roof_displacement']*100:.2f} cm")
    print(f"  Max drift:         {metrics['max_drift']*100:.2f} cm")
    print(f"  DCR:               {metrics['DCR']:.4f}")
    print(f"  Peak force:        {metrics['peak_force_kN']:.1f} kN")
    print(f"  Mean force:        {metrics['mean_force_kN']:.1f} kN")

    return metrics


def save_hybrid_model_as_wrapper(model1_path, model2_path, output_path, strategy='adaptive'):
    """
    Save hybrid model configuration

    Note: Since we can't easily save a custom class as a .zip file in Stable Baselines3 format,
    we'll create a Python module that can be imported and used.
    """
    print(f"\nSaving hybrid model configuration to {output_path}...")

    # Create wrapper code
    wrapper_code = f'''"""
Auto-generated hybrid model combining two RL models

Model 1: {model1_path.name} (Peak displacement optimization)
Model 2: {model2_path.name} (DCR optimization)
Strategy: {strategy}
"""

from create_hybrid_model import HybridRLModel
from pathlib import Path

# Model paths
MODEL1_PATH = Path(__file__).parent / "{model1_path.name}"
MODEL2_PATH = Path(__file__).parent / "{model2_path.name}"

# Create hybrid instance
hybrid_model = HybridRLModel(MODEL1_PATH, MODEL2_PATH, strategy="{strategy}")

def predict(observation, deterministic=True):
    """Predict action using hybrid model"""
    return hybrid_model.predict(observation, deterministic)

def reset_episode(max_steps=2000):
    """Reset for new episode"""
    hybrid_model.reset_episode(max_steps)
'''

    with open(output_path, 'w') as f:
        f.write(wrapper_code)

    print(f"  Saved hybrid wrapper to {output_path}")
    print(f"  Usage: import hybrid_model; action, _ = hybrid_model.predict(obs)")


def compare_all_strategies(model1_path, model2_path, earthquake_file):
    """
    Compare all combination strategies

    Args:
        model1_path: Path to old model
        model2_path: Path to new model
        earthquake_file: Test earthquake
    """
    print("\n" + "=" * 70)
    print("COMPARING ALL HYBRID STRATEGIES")
    print("=" * 70)

    strategies = ['adaptive', 'weighted_avg', 'max_response']
    results = {}

    for strategy in strategies:
        metrics = test_hybrid_model(model1_path, model2_path, earthquake_file, strategy)
        results[strategy] = metrics

    # Also test individual models for comparison
    print("\n" + "=" * 70)
    print("BASELINE: MODEL 1 ONLY (Old Model)")
    print("=" * 70)

    model1 = SAC.load(model1_path)
    env = make_simple_4d_env(earthquake_file)
    obs, info = env.reset()

    done = False
    while not done:
        action, _ = model1.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    metrics1 = env.get_episode_metrics()
    print(f"  Peak displacement: {metrics1['peak_roof_displacement']*100:.2f} cm")
    print(f"  DCR:               {metrics1['DCR']:.4f}")
    results['model1_only'] = metrics1

    print("\n" + "=" * 70)
    print("BASELINE: MODEL 2 ONLY (New Model)")
    print("=" * 70)

    model2 = SAC.load(model2_path)
    env = make_simple_4d_env(earthquake_file)
    obs, info = env.reset()

    done = False
    while not done:
        action, _ = model2.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    metrics2 = env.get_episode_metrics()
    print(f"  Peak displacement: {metrics2['peak_roof_displacement']*100:.2f} cm")
    print(f"  DCR:               {metrics2['DCR']:.4f}")
    results['model2_only'] = metrics2

    # Summary comparison
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"\n{'Strategy':<20} {'Peak (cm)':<12} {'DCR':<8} {'Score':<8}")
    print("-" * 50)

    for name, metrics in results.items():
        peak_cm = metrics['peak_roof_displacement'] * 100
        dcr = metrics['DCR']
        # Combined score: lower peak + lower DCR = better
        # Normalize: peak/50 + DCR/1.5
        score = peak_cm / 50.0 + dcr / 1.5
        print(f"{name:<20} {peak_cm:<12.2f} {dcr:<8.4f} {score:<8.2f}")

    # Find best strategy
    best_strategy = min(results.items(),
                       key=lambda x: x[1]['peak_roof_displacement'] + x[1]['DCR']/100)

    print(f"\nBest strategy: {best_strategy[0].upper()}")
    print(f"  Peak: {best_strategy[1]['peak_roof_displacement']*100:.2f} cm")
    print(f"  DCR:  {best_strategy[1]['DCR']:.4f}")

    return results, best_strategy[0]


if __name__ == "__main__":
    # Paths
    MODEL1_PATH = Path("../models/rl_cl_final_robust.zip")  # Old model (good peak)
    MODEL2_PATH = Path("../models/rl_cl_dcr_train_5_final.zip")   # New model (good DCR) - Latest training

    # Test earthquake (M7.4 to really test them)
    EARTHQUAKE = Path("../../matlab/datasets/training_set/TRAIN_M7.4_PGA0.75g_variant1.csv")

    # Check files exist
    if not MODEL1_PATH.exists():
        print(f"ERROR: Model 1 not found: {MODEL1_PATH}")
        sys.exit(1)

    if not MODEL2_PATH.exists():
        print(f"ERROR: Model 2 not found: {MODEL2_PATH}")
        print(f"\nPlease update MODEL2_PATH to point to your new model!")
        print(f"Available models:")
        models_dir = Path("../models")
        for model in sorted(models_dir.glob("*.zip")):
            print(f"  - {model.name}")
        sys.exit(1)

    if not EARTHQUAKE.exists():
        print(f"ERROR: Earthquake not found: {EARTHQUAKE}")
        sys.exit(1)

    # Compare all strategies
    results, best_strategy = compare_all_strategies(MODEL1_PATH, MODEL2_PATH, EARTHQUAKE)

    # Save best hybrid model
    output_path = Path("../models/hybrid_rl_model_best.py")
    save_hybrid_model_as_wrapper(MODEL1_PATH, MODEL2_PATH, output_path, strategy=best_strategy)

    print("\n" + "=" * 70)
    print("HYBRID MODEL CREATION COMPLETE!")
    print("=" * 70)
    print(f"\nBest strategy: {best_strategy}")
    print(f"Saved to: {output_path}")
    print(f"\nTo use in API, update RLCLController to load hybrid model")

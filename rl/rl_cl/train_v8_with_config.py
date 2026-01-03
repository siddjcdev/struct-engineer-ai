"""
v8 Training - PPO with Configuration Module
===========================================

Clean training script using ppo_config.py for all configurations.

This script demonstrates how to use the centralized configuration
for maintainable, well-documented PPO training.

Usage: python train_v8_with_config.py

Author: Siddharth
Date: January 2026
"""

import numpy as np
import os
import glob
import random
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from tmd_environment_adaptive_reward import make_improved_tmd_env
from ppo_config import V8PPOConfig


def make_env_factory(train_files, force_limit):
    """
    Factory function to create environment makers

    Args:
        train_files: List of training earthquake file paths
        force_limit: Maximum control force (N)

    Returns:
        Function that creates a monitored environment
    """
    def _make_env():
        eq_file = random.choice(train_files)
        env = make_improved_tmd_env(eq_file, max_force=force_limit)
        return Monitor(env)
    return _make_env


def create_parallel_envs(train_files, force_limit, n_envs=4):
    """
    Create parallel training environments

    Args:
        train_files: List of training earthquake files
        force_limit: Maximum control force (N)
        n_envs: Number of parallel environments

    Returns:
        Vectorized environment (SubprocVecEnv or DummyVecEnv)
    """
    env_fns = [make_env_factory(train_files, force_limit) for _ in range(n_envs)]

    try:
        env = SubprocVecEnv(env_fns)
        print(f"   ✓ Using {n_envs} parallel environments (SubprocVecEnv)\n")
    except Exception as e:
        print(f"   ⚠ SubprocVecEnv failed ({e}), falling back to DummyVecEnv")
        env = DummyVecEnv(env_fns)
        print(f"   ✓ Using {n_envs} sequential environments (DummyVecEnv - slower)\n")

    return env


def create_ppo_model(env, stage_config, device):
    """
    Create PPO model with stage-specific hyperparameters

    Args:
        env: Training environment
        stage_config: Stage configuration dict
        device: 'cuda' or 'cpu'

    Returns:
        PPO model instance
    """
    from ppo_config import PPOHyperparameters

    # Get base configuration
    base_config = PPOHyperparameters.get_base_config(device)

    # Add stage-specific parameters
    model = PPO(
        base_config['policy'],
        env,
        learning_rate=stage_config['learning_rate'],
        n_steps=stage_config['n_steps'],
        batch_size=base_config['batch_size'],
        n_epochs=base_config['n_epochs'],
        gamma=base_config['gamma'],
        gae_lambda=base_config['gae_lambda'],
        clip_range=base_config['clip_range'],
        clip_range_vf=base_config['clip_range_vf'],
        ent_coef=stage_config['ent_coef'],
        vf_coef=base_config['vf_coef'],
        max_grad_norm=base_config['max_grad_norm'],
        policy_kwargs=base_config['policy_kwargs'],
        verbose=base_config['verbose'],
        device=base_config['device']
    )

    return model


def update_model_hyperparameters(model, stage_config):
    """
    Update model hyperparameters for new stage

    Args:
        model: Existing PPO model
        stage_config: New stage configuration dict
    """
    # Update learning rate and entropy coefficient
    model.learning_rate = stage_config['learning_rate']
    model.ent_coef = stage_config['ent_coef']

    # Recreate optimizer with new learning rate
    model.policy.optimizer = model.policy.optimizer_class(
        model.policy.parameters(),
        lr=stage_config['learning_rate'],
        **model.policy.optimizer_kwargs
    )


def test_on_held_out(model, test_file, force_limit):
    """
    Test model on held-out earthquake

    Args:
        model: Trained PPO model
        test_file: Path to test earthquake file
        force_limit: Maximum control force

    Returns:
        Peak displacement in cm
    """
    test_env = make_improved_tmd_env(test_file, max_force=force_limit)
    obs, _ = test_env.reset()
    done = False
    peak = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = test_env.step(action)
        peak = max(peak, abs(info['roof_displacement']))
        done = done or truncated

    peak_cm = peak * 100
    return peak_cm


def run_final_evaluation(model, config):
    """
    Run comprehensive evaluation on all held-out earthquakes

    Args:
        model: Trained PPO model
        config: V8PPOConfig instance

    Returns:
        List of result dictionaries
    """
    print(f"\n{'='*70}")
    print(f"  FINAL TEST ON ALL HELD-OUT EARTHQUAKES")
    print(f"{'='*70}\n")

    results = []

    for magnitude, test_file in config.data.TEST_FILES.items():
        test_env = make_improved_tmd_env(test_file, max_force=150000)
        obs, _ = test_env.reset()
        done = False
        peak = 0
        total_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = test_env.step(action)
            peak = max(peak, abs(info['roof_displacement']))
            total_reward += reward
            done = done or truncated

        peak_cm = peak * 100
        episode_metrics = test_env.get_episode_metrics()
        dcr = episode_metrics.get('dcr', 0.0)

        uncont = config.data.UNCONTROLLED_BASELINES[magnitude]
        improvement = 100 * (uncont - peak_cm) / uncont

        results.append({
            'magnitude': magnitude,
            'peak_cm': peak_cm,
            'dcr': dcr,
            'improvement': improvement
        })

        status = "✓" if improvement > 0 else "✗"
        print(f"   {magnitude}:")
        print(f"      Peak disp: {peak_cm:.2f} cm")
        print(f"      Uncontrolled: {uncont:.2f} cm")
        print(f"      Improvement: {improvement:+.1f}% {status}")
        print(f"      DCR: {dcr:.2f}\n")

    return results


def print_comparison(results, config):
    """
    Print comparison with previous model versions

    Args:
        results: List of evaluation results
        config: V8PPOConfig instance
    """
    print("="*70)
    print("  COMPARISON: v8-PPO-CONFIG vs v7-SAC vs v6 vs v5")
    print("="*70)

    for r in results:
        mag = r['magnitude']
        v8_peak = r['peak_cm']

        print(config.baselines.get_comparison_summary(mag, v8_peak))

    print("\n" + "="*70 + "\n")


def train_v8_with_config():
    """
    Main training function using centralized configuration

    This function orchestrates the entire training process using
    the V8PPOConfig class for all settings.
    """
    # Initialize configuration
    config = V8PPOConfig()

    # Print full configuration
    config.print_full_config()

    # Extract commonly used values
    device = config.device_config['device']
    gpu_name = config.device_config.get('gpu_name')
    stages = config.curriculum.STAGES
    n_envs = config.hyperparameters.N_ENVS

    # Find training files
    print("📁 Locating training files...")
    train_files = {}
    for magnitude in ['M4.5', 'M5.7', 'M7.4', 'M8.4']:
        pattern = f"{config.data.TRAIN_DIR}/TRAIN_{magnitude}_*.csv"
        files = sorted(glob.glob(pattern))

        if not files:
            print(f"   ❌ {magnitude}: No training files found!")
            print(f"\n   ERROR: Run generate_training_earthquakes_v2.py first!")
            return None

        train_files[magnitude] = files
        print(f"   ✓ {magnitude}: {len(files)} variants")

    print()

    # Verify test files
    print("📁 Verifying test files...")
    for mag, file in config.data.TEST_FILES.items():
        exists = "✓" if os.path.exists(file) else "❌"
        print(f"   {exists} {mag}: {os.path.basename(file)}")
    print()

    # Create output directory
    os.makedirs(config.saving.OUTPUT_DIR, exist_ok=True)

    # Training loop
    start_time = datetime.now()
    model = None

    for stage_idx, stage in enumerate(stages):
        stage_num = stage_idx + 1
        magnitude = stage['magnitude']
        force_limit = stage['force_limit']
        timesteps = stage['timesteps']
        available_files = train_files[magnitude]

        print(f"\n{'='*70}")
        print(f"  STAGE {stage_num}/{len(stages)}: {stage['name']}")
        print(f"{'='*70}")
        print(f"\n   {stage['description']}")
        print(f"\n   Training variants: {len(available_files)}")
        print(f"   Hyperparameters:")
        print(f"      n_steps: {stage['n_steps']}")
        print(f"      learning_rate: {stage['learning_rate']:.0e}")
        print(f"      ent_coef: {stage['ent_coef']}")
        print(f"      timesteps: {timesteps:,}")
        print(f"      force_limit: {force_limit:,} N\n")

        # Create parallel environments
        env = create_parallel_envs(available_files, force_limit, n_envs)

        # Create or update model
        if model is None:
            print(f"🤖 Creating PPO model...")
            print(f"   Device: {device}")
            if gpu_name:
                print(f"   GPU: {gpu_name}")
            print()

            model = create_ppo_model(env, stage, device)
        else:
            print(f"🔄 Continuing from Stage {stage_num-1}...")
            print(f"   Updating hyperparameters for Stage {stage_num}...\n")

            # Close old environment and set new one
            env_old = model.get_env()
            model.set_env(env)
            env_old.close()

            # Update hyperparameters
            update_model_hyperparameters(model, stage)

        # Train
        print(f"🚀 Training Stage {stage_num}...")
        model.learn(
            total_timesteps=timesteps,
            reset_num_timesteps=False,
            progress_bar=True
        )

        # Save checkpoint
        save_path = config.saving.get_stage_checkpoint_path(stage_num, force_limit)
        model.save(save_path)
        print(f"\n💾 Saved: {save_path}")

        # Test on held-out earthquake
        test_file = config.data.TEST_FILES[magnitude]
        print(f"\n📊 Testing on HELD-OUT test earthquake...")
        print(f"   Test file: {os.path.basename(test_file)}")

        peak_cm = test_on_held_out(model, test_file, force_limit)
        print(f"   Peak displacement: {peak_cm:.2f} cm")
        print(f"\n✅ Stage {stage_num} complete!\n")

        # Close environments
        env.close()

    # Final model save
    training_time = datetime.now() - start_time
    final_path = config.saving.get_final_model_path()
    model.save(final_path)

    # Print training summary
    print("="*70)
    print("  🎉 TRAINING COMPLETE!")
    print("="*70)
    print(f"\n   Total training time: {training_time}")
    print(f"   Training device: {device.upper()}")
    if gpu_name:
        print(f"   GPU: {gpu_name}")
    print(f"   Final model: {final_path}")
    print(f"\n   Configuration:")
    print(f"   • PPO with optimized hyperparameters")
    print(f"   • {n_envs} parallel environments")
    print(f"   • Adaptive n_steps (1024 → 4096)")
    print(f"   • Curriculum learning rate (3e-4 → 5e-5)")
    print(f"   • Entropy annealing (0.02 → 0.001)")
    print(f"   • Adaptive reward scaling (3×, 7×, 4×, 3×)")
    print(f"   • Total timesteps: {config.curriculum.get_total_timesteps():,}")
    print(f"   • Effective samples: {config.curriculum.get_effective_samples(n_envs):,}")

    # Final evaluation
    results = run_final_evaluation(model, config)

    # Print comparison with previous versions
    print_comparison(results, config)

    return model


if __name__ == "__main__":
    print("\n🚀 Starting v8 PPO Training with Configuration Module...\n")

    model = train_v8_with_config()

    if model is not None:
        print("✅ Training complete! Model ready for deployment.")
        print("\nNext steps:")
        print("  1. Evaluate on additional test cases")
        print("  2. Deploy to production API")
        print("  3. Compare with v7-SAC baseline")
    else:
        print("❌ Training failed - check training files and configuration.")

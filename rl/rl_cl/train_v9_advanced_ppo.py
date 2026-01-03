"""
v9 Advanced PPO Training - Refined Hyperparameters
==================================================

This script implements advanced PPO optimizations based on the suggestions:
1. Larger n_steps (2048-8192) for reduced variance
2. Balanced batch_size (256-512) and n_epochs (10-20)
3. Smoother learning rate transitions with cosine annealing
4. Refined entropy scheduling
5. Deeper network architecture (3 layers)
6. Tighter value function clipping (0.15)

Expected improvements over v8:
- More stable learning on M7.4 and M8.4
- Better generalization from reduced variance
- Smoother convergence with cosine LR decay

Usage: python train_v9_advanced_ppo.py

Author: Siddharth
Date: January 2026
"""

import numpy as np
import os
import glob
import random
from datetime import datetime
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from tmd_environment_adaptive_reward import make_improved_tmd_env
from ppo_config_v9_advanced import (
    V9AdvancedConfig,
    V9PPOHyperparameters,
    V9CurriculumStages
)


def make_env_factory(train_files, force_limit):
    """Factory function to create environment makers"""
    def _make_env():
        eq_file = random.choice(train_files)
        env = make_improved_tmd_env(eq_file, max_force=force_limit)
        return Monitor(env)
    return _make_env


def create_parallel_envs(train_files, force_limit, n_envs=4):
    """Create parallel training environments"""
    env_fns = [make_env_factory(train_files, force_limit) for _ in range(n_envs)]

    try:
        env = SubprocVecEnv(env_fns)
        print(f"   ✓ Using {n_envs} parallel environments (SubprocVecEnv)\n")
    except Exception as e:
        print(f"   ⚠ SubprocVecEnv failed, falling back to DummyVecEnv")
        env = DummyVecEnv(env_fns)
        print(f"   ✓ Using {n_envs} sequential environments (DummyVecEnv - slower)\n")

    return env


def create_entropy_schedule(stage):
    """
    Create entropy coefficient schedule if enabled

    Args:
        stage: Stage configuration dict

    Returns:
        Entropy coefficient (float or schedule function)
    """
    if stage.get('ent_schedule', False):
        initial_ent = stage['ent_coef']
        final_ent = stage.get('final_ent', initial_ent * 0.1)

        def ent_schedule(progress_remaining):
            # Linear decay from initial to final
            return final_ent + (initial_ent - final_ent) * progress_remaining

        return ent_schedule
    else:
        return stage['ent_coef']


def create_v9_ppo_model(env, stage, device):
    """
    Create PPO model with v9 advanced configuration

    Args:
        env: Training environment
        stage: Stage configuration dict
        device: 'cuda' or 'cpu'

    Returns:
        PPO model instance
    """
    # Get learning rate schedule
    lr_schedule = V9CurriculumStages.get_learning_rate_schedule(stage)

    # Get entropy schedule
    ent_schedule = create_entropy_schedule(stage)

    # Get advanced policy kwargs
    policy_kwargs = V9PPOHyperparameters.get_policy_kwargs(use_advanced_arch=True)

    print(f"   Creating model with advanced configuration:")
    print(f"      Network: {policy_kwargs['net_arch']}")
    print(f"      n_steps: {stage['n_steps']}")
    print(f"      batch_size: {stage['batch_size']}")
    print(f"      n_epochs: {stage['n_epochs']}")

    if stage.get('use_lr_schedule'):
        print(f"      learning_rate: {stage['learning_rate']:.0e} → {stage['final_lr']:.0e} (cosine)")
    else:
        print(f"      learning_rate: {stage['learning_rate']:.0e} (fixed)")

    if stage.get('ent_schedule'):
        print(f"      ent_coef: {stage['ent_coef']} → {stage['final_ent']} (annealing)")
    else:
        print(f"      ent_coef: {stage['ent_coef']} (fixed)")

    print()

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=lr_schedule,
        n_steps=stage['n_steps'],
        batch_size=stage['batch_size'],
        n_epochs=stage['n_epochs'],
        gamma=V9PPOHyperparameters.GAMMA,
        gae_lambda=V9PPOHyperparameters.GAE_LAMBDA,
        clip_range=V9PPOHyperparameters.CLIP_RANGE,
        clip_range_vf=V9PPOHyperparameters.CLIP_RANGE_VF,
        ent_coef=ent_schedule,
        vf_coef=V9PPOHyperparameters.VF_COEF,
        max_grad_norm=V9PPOHyperparameters.MAX_GRAD_NORM,
        policy_kwargs=policy_kwargs,
        verbose=V9PPOHyperparameters.VERBOSE,
        device=device
    )

    return model


def update_model_for_new_stage(model, env, stage):
    """
    Update model for new curriculum stage

    Args:
        model: Existing PPO model
        env: New training environment
        stage: New stage configuration
    """
    print(f"   Updating model for new stage:")

    # Update environment
    env_old = model.get_env()
    model.set_env(env)
    env_old.close()

    # Update learning rate schedule
    lr_schedule = V9CurriculumStages.get_learning_rate_schedule(stage)
    if callable(lr_schedule):
        print(f"      learning_rate: Cosine schedule ({stage['learning_rate']:.0e} → {stage['final_lr']:.0e})")
        model.learning_rate = lr_schedule
    else:
        print(f"      learning_rate: {lr_schedule:.0e} (fixed)")
        model.learning_rate = lr_schedule

    # Update entropy coefficient
    ent_schedule = create_entropy_schedule(stage)
    if callable(ent_schedule):
        print(f"      ent_coef: Annealing ({stage['ent_coef']} → {stage['final_ent']})")
        model.ent_coef = ent_schedule
    else:
        print(f"      ent_coef: {ent_schedule} (fixed)")
        model.ent_coef = ent_schedule

    # Recreate optimizer with new learning rate
    # Note: If using schedule, get initial value
    initial_lr = lr_schedule(1.0) if callable(lr_schedule) else lr_schedule
    model.policy.optimizer = model.policy.optimizer_class(
        model.policy.parameters(),
        lr=initial_lr,
        **model.policy.optimizer_kwargs
    )

    print()


def test_on_earthquake(model, test_file, force_limit):
    """Test model on single earthquake"""
    test_env = make_improved_tmd_env(test_file, max_force=force_limit)
    obs, _ = test_env.reset()
    done = False
    peak = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = test_env.step(action)
        peak = max(peak, abs(info['roof_displacement']))
        done = done or truncated

    return peak * 100  # Convert to cm


def train_v9_advanced():
    """Main training function with v9 advanced configuration"""

    # Auto-detect device
    gpu_name = None
    if torch.cuda.is_available():
        device = 'cuda'
        gpu_name = torch.cuda.get_device_name(0)
        print("="*70)
        print("  V9 ADVANCED PPO TRAINING (GPU ACCELERATED)")
        print("="*70)
        print(f"\n🚀 GPU: {gpu_name}")
        print(f"   Device: {device}\n")
    else:
        device = 'cpu'
        print("="*70)
        print("  V9 ADVANCED PPO TRAINING")
        print("="*70)
        print("\n⚠️  No GPU detected - using CPU\n")

    # Print improvements summary
    V9AdvancedConfig.print_improvements_summary()

    # Print curriculum
    V9AdvancedConfig.curriculum.print_curriculum_summary()

    # Dataset paths
    train_dir = "../../matlab/datasets/training_set_v2"
    test_files = {
        'M4.5': "../../matlab/datasets/PEER_small_M4.5_PGA0.25g.csv",
        'M5.7': "../../matlab/datasets/PEER_moderate_M5.7_PGA0.35g.csv",
        'M7.4': "../../matlab/datasets/PEER_high_M7.4_PGA0.75g.csv",
        'M8.4': "../../matlab/datasets/PEER_insane_M8.4_PGA0.9g.csv"
    }

    # Find training files
    print("📁 Locating training files...")
    train_files = {}
    for magnitude in ['M4.5', 'M5.7', 'M7.4', 'M8.4']:
        pattern = f"{train_dir}/TRAIN_{magnitude}_*.csv"
        files = sorted(glob.glob(pattern))

        if not files:
            print(f"   ❌ {magnitude}: No training files found!")
            return None

        train_files[magnitude] = files
        print(f"   ✓ {magnitude}: {len(files)} variants")

    print()

    # Create output directory
    output_dir = "models/rl_v9_advanced"
    os.makedirs(output_dir, exist_ok=True)

    # Training loop
    start_time = datetime.now()
    model = None
    n_envs = V9PPOHyperparameters.N_ENVS
    stages = V9CurriculumStages.STAGES

    for stage_idx, stage in enumerate(stages):
        stage_num = stage_idx + 1
        magnitude = stage['magnitude']
        force_limit = stage['force_limit']
        timesteps = stage['timesteps']
        available_files = train_files[magnitude]

        print(f"\n{'='*70}")
        print(f"  STAGE {stage_num}/{len(stages)}: {stage['name']}")
        print(f"{'='*70}")
        print(f"\n   {stage['description']}\n")
        print(f"   Training variants: {len(available_files)}")
        print(f"   Timesteps: {timesteps:,}\n")

        # Create parallel environments
        env = create_parallel_envs(available_files, force_limit, n_envs)

        # Create or update model
        if model is None:
            print(f"🤖 Creating v9 Advanced PPO model...")
            print(f"   Device: {device}\n")
            model = create_v9_ppo_model(env, stage, device)
        else:
            print(f"🔄 Continuing from Stage {stage_num-1}...\n")
            update_model_for_new_stage(model, env, stage)

        # Train
        print(f"🚀 Training Stage {stage_num}...")
        model.learn(
            total_timesteps=timesteps,
            reset_num_timesteps=False,
            progress_bar=True
        )

        # Save checkpoint
        save_path = f"{output_dir}/stage{stage_num}_{force_limit//1000}kN.zip"
        model.save(save_path)
        print(f"\n💾 Saved: {save_path}")

        # Test on held-out earthquake
        test_file = test_files[magnitude]
        print(f"\n📊 Testing on HELD-OUT test earthquake...")
        print(f"   Test file: {os.path.basename(test_file)}")

        peak_cm = test_on_earthquake(model, test_file, force_limit)
        print(f"   Peak displacement: {peak_cm:.2f} cm")
        print(f"\n✅ Stage {stage_num} complete!\n")

        # Close environments
        env.close()

    # Final model save
    training_time = datetime.now() - start_time
    final_path = f"{output_dir}/final_v9_advanced.zip"
    model.save(final_path)

    # Print summary
    print("="*70)
    print("  🎉 TRAINING COMPLETE!")
    print("="*70)
    print(f"\n   Total training time: {training_time}")
    print(f"   Training device: {device.upper()}")
    if gpu_name:
        print(f"   GPU: {gpu_name}")
    print(f"   Final model: {final_path}")
    print(f"\n   V9 Advanced Features:")
    print(f"   • Deeper network: [256, 256, 256]")
    print(f"   • Larger n_steps: 2048-8192 (reduced variance)")
    print(f"   • Balanced batch_size: 256-512")
    print(f"   • Optimized n_epochs: 10-20")
    print(f"   • Cosine LR annealing (smoother decay)")
    print(f"   • Refined entropy scheduling")
    print(f"   • Tighter value clipping: 0.15")
    print(f"   • Total timesteps: {V9CurriculumStages.get_total_timesteps():,}")

    # Final evaluation
    print(f"\n{'='*70}")
    print(f"  FINAL EVALUATION ON ALL HELD-OUT EARTHQUAKES")
    print(f"{'='*70}\n")

    uncontrolled_baselines = {
        'M4.5': 21.02,
        'M5.7': 46.02,
        'M7.4': 235.55,
        'M8.4': 357.06
    }

    v8_results = {
        'M4.5': 20.72,  # v7-SAC result (best so far)
        'M5.7': 46.45,
        'M7.4': 219.30,  # Best SAC result
        'M8.4': 363.36
    }

    results = []
    for magnitude, test_file in test_files.items():
        peak_cm = test_on_earthquake(model, test_file, 150000)

        uncont = uncontrolled_baselines[magnitude]
        improvement = 100 * (uncont - peak_cm) / uncont

        results.append({
            'magnitude': magnitude,
            'peak_cm': peak_cm,
            'improvement': improvement
        })

        status = "✓" if improvement > 0 else "✗"
        v8_peak = v8_results[magnitude]
        delta_v8 = v8_peak - peak_cm
        v8_status = "🏆 IMPROVED" if delta_v8 > 0 else "⚠️ SIMILAR" if abs(delta_v8) < 0.5 else "❌ WORSE"

        print(f"   {magnitude}:")
        print(f"      v9-Advanced: {peak_cm:.2f} cm")
        print(f"      v8-Baseline: {v8_peak:.2f} cm")
        print(f"      Δ from v8: {delta_v8:+.2f} cm {v8_status}")
        print(f"      Uncontrolled: {uncont:.2f} cm")
        print(f"      Improvement: {improvement:+.1f}% {status}\n")

    print("="*70 + "\n")

    return model


if __name__ == "__main__":
    print("\n🚀 Starting v9 Advanced PPO Training...\n")

    model = train_v9_advanced()

    if model is not None:
        print("✅ v9 Advanced training complete!")
        print("\nKey improvements implemented:")
        print("  ✓ Larger n_steps (2× increase)")
        print("  ✓ Balanced batch_size and n_epochs")
        print("  ✓ Cosine LR annealing")
        print("  ✓ Deeper network architecture")
        print("  ✓ Refined value function clipping")
        print("\nExpected benefits:")
        print("  • Reduced variance in advantage estimates")
        print("  • Smoother convergence")
        print("  • Better generalization on M7.4/M8.4")
    else:
        print("❌ Training failed - check training files.")

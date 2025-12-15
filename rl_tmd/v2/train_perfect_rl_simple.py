"""
SIMPLIFIED PERFECT RL TRAINING
==============================

Curriculum learning without complex early stopping
Guaranteed to work!

Usage: python train_perfect_rl_simple.py --earthquakes <files>
"""

import numpy as np
import torch
import os
from datetime import datetime
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from improved_tmd_environment import make_improved_tmd_env
import matplotlib.pyplot as plt


def train_simple(earthquake_files):
    """
    Simplified training with curriculum, no complex callbacks
    """
    
    print("="*70)
    print("  SIMPLIFIED PERFECT RL TRAINING")
    print("="*70)
    print("\n🎯 Curriculum Plan:")
    
    # Curriculum stages
    stages = [
        {'force_limit': 50000,  'timesteps': 150000, 'name': '50 kN (gentle)'},
        {'force_limit': 100000, 'timesteps': 150000, 'name': '100 kN (moderate)'},
        {'force_limit': 150000, 'timesteps': 200000, 'name': '150 kN (full)'},
    ]
    
    for i, stage in enumerate(stages, 1):
        print(f"   Stage {i}: {stage['name']} - {stage['timesteps']:,} steps")
    
    # Create directories
    os.makedirs("simple_rl_models", exist_ok=True)
    
    # Training
    start_time = datetime.now()
    model = None
    
    for stage_idx, stage in enumerate(stages):
        stage_num = stage_idx + 1
        force_limit = stage['force_limit']
        timesteps = stage['timesteps']
        
        print(f"\n{'='*70}")
        print(f"  STAGE {stage_num}: {stage['name']}")
        print(f"{'='*70}\n")
        
        # Create environment
        def make_env(eq_file, force_lim):
            env = make_improved_tmd_env(eq_file, max_force=force_lim)
            env = Monitor(env)
            return env
        
        # Use first earthquake file for training
        env = DummyVecEnv([lambda: make_env(earthquake_files[0], force_limit)])
        
        # Create or update model
        if model is None:
            print(f"🤖 Creating SAC model...")
            model = SAC(
                "MlpPolicy",
                env,
                learning_rate=3e-4,
                buffer_size=100_000,
                batch_size=256,
                gamma=0.99,
                tau=0.005,
                ent_coef='auto',
                policy_kwargs=dict(net_arch=[256, 256]),
                verbose=1,
                device='auto'
            )
        else:
            print(f"🔄 Updating environment...")
            model.set_env(env)
        
        # Train
        print(f"🚀 Training...")
        model.learn(
            total_timesteps=timesteps,
            reset_num_timesteps=False,
            progress_bar=True
        )
        
        # Save
        save_path = f"simple_rl_models/stage{stage_num}_{force_limit//1000}kN.zip"
        model.save(save_path)
        print(f"💾 Saved: {save_path}")
        
        # Quick test
        print(f"\n📊 Testing stage {stage_num}...")
        test_env = make_improved_tmd_env(earthquake_files[0], max_force=force_limit)
        obs, _ = test_env.reset()
        done = False
        truncated = False
        peak = 0
        
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = test_env.step(action)
            peak = max(peak, abs(info['roof_displacement']))
        
        peak_cm = peak * 100
        print(f"   Peak displacement: {peak_cm:.2f} cm")
        print(f"✅ Stage {stage_num} complete!\n")
    
    # Final
    training_time = datetime.now() - start_time
    final_path = "simple_rl_models/perfect_rl_final.zip"
    model.save(final_path)
    
    print("="*70)
    print("  🎉 TRAINING COMPLETE!")
    print("="*70)
    print(f"\n   Total time: {training_time}")
    print(f"   Final model: {final_path}")
    print(f"\n   Test with:")
    print(f"   python test_rl_model.py --model {final_path} --earthquake <file>")
    print("="*70 + "\n")
    
    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Simplified Perfect RL Training')
    parser.add_argument('--earthquakes', nargs='+', required=True,
                       help='Earthquake CSV files')
    
    args = parser.parse_args()
    
    print("\n🚀 Starting Simplified Perfect RL Training...\n")
    model = train_simple(args.earthquakes)
    print("\n✅ All done!")
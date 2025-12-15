"""
PERFECT RL TRAINING - ALL 4 FIXES APPLIED
=========================================

Implements:
1. ✅ Early Stopping with Validation
2. ✅ Better Multi-Objective Reward Function
3. ✅ Curriculum Learning (Progressive Force Limits)
4. ✅ Regularization (Smoothness + Acceleration)

Author: Siddharth
Date: December 2025
"""

import numpy as np
import torch
import os
from datetime import datetime
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
from improved_tmd_environment import ImprovedTMDBuildingEnv, make_improved_tmd_env


# ================================================================
# IMPROVED CONFIGURATION
# ================================================================

class PerfectRLConfig:
    """Configuration for perfect RL training"""
    
    # Curriculum stages (progressive force limits)
    curriculum_stages = [
        {'force_limit': 50000,  'timesteps': 150000, 'name': 'Stage 1: Basic Control (50 kN)'},
        {'force_limit': 100000, 'timesteps': 150000, 'name': 'Stage 2: Moderate Control (100 kN)'},
        {'force_limit': 150000, 'timesteps': 200000, 'name': 'Stage 3: Advanced Control (150 kN)'},
    ]
    
    # Early stopping
    validation_freq = 5000      # Validate every 5k steps
    patience = 20               # Stop if no improvement for 20 validations (100k steps)
    min_improvement = 0.01      # Minimum improvement to count (1%)
    
    # SAC hyperparameters
    learning_rate = 3e-4
    batch_size = 256
    buffer_size = 100_000
    gamma = 0.99
    tau = 0.005
    ent_coef = 'auto'
    policy_layers = [256, 256]
    
    # Paths
    model_save_dir = "perfect_rl_models"
    log_dir = "perfect_rl_logs"
    tensorboard_log = "perfect_rl_tensorboard"


# ================================================================
# EARLY STOPPING CALLBACK
# ================================================================

class EarlyStoppingCallback(BaseCallback):
    """
    Callback for early stopping based on validation performance
    """
    
    def __init__(
        self,
        validation_env,
        validation_freq: int = 5000,
        patience: int = 20,
        min_improvement: float = 0.01,
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.validation_env = validation_env
        self.validation_freq = validation_freq
        self.patience = patience
        self.min_improvement = min_improvement
        
        self.best_mean_reward = -np.inf
        self.best_peak_disp = np.inf
        self.patience_counter = 0
        self.validation_history = []
        
        self.episode_rewards = []
        self.episode_peaks = []
    
    
    def _on_step(self) -> bool:
        # Track training episodes
        if self.locals.get('dones', [False])[0]:
            info = self.locals.get('infos', [{}])[0]
            if 'episode' in self.locals:
                ep_info = self.locals['episode'][0]
                self.episode_rewards.append(ep_info['r'])
            
            if 'peak_displacement' in info:
                self.episode_peaks.append(info['peak_displacement'] * 100)
        
        # Validation check
        if self.num_timesteps % self.validation_freq == 0:
            val_reward, val_peak = self._validate()
            
            # Check for improvement
            improved = False
            if val_peak < self.best_peak_disp * (1 - self.min_improvement):
                self.best_peak_disp = val_peak
                self.best_mean_reward = val_reward
                improved = True
                self.patience_counter = 0
                
                # Save best model
                model_path = os.path.join(
                    PerfectRLConfig.model_save_dir,
                    f"best_model_{self.num_timesteps}steps.zip"
                )
                self.model.save(model_path)
                
                if self.verbose > 0:
                    print(f"\n✅ NEW BEST at {self.num_timesteps} steps!")
                    print(f"   Peak: {val_peak:.2f} cm, Reward: {val_reward:.2f}")
                    print(f"   Saved: {model_path}")
            else:
                self.patience_counter += 1
                
                if self.verbose > 0:
                    print(f"\n⏸️  Validation at {self.num_timesteps} steps:")
                    print(f"   Peak: {val_peak:.2f} cm (best: {self.best_peak_disp:.2f})")
                    print(f"   Patience: {self.patience_counter}/{self.patience}")
            
            # Store history
            self.validation_history.append({
                'timestep': self.num_timesteps,
                'peak_disp': val_peak,
                'mean_reward': val_reward,
                'improved': improved
            })
            
            # Check early stopping
            if self.patience_counter >= self.patience:
                if self.verbose > 0:
                    print(f"\n🛑 EARLY STOPPING triggered at {self.num_timesteps} steps")
                    print(f"   No improvement for {self.patience} validations")
                    print(f"   Best peak: {self.best_peak_disp:.2f} cm")
                return False  # Stop training
        
        return True
    
    
    def _validate(self, n_episodes: int = 3) -> tuple:
        """Run validation episodes"""
        rewards = []
        peaks = []
        
        for _ in range(n_episodes):
            obs = self.validation_env.reset()
            done = False
            episode_reward = 0
            peak_disp = 0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                # Vectorized env returns: obs, rewards, dones, infos (4 values)
                obs, reward, done, info = self.validation_env.step(action)
                
                # Extract from vectorized format
                if isinstance(reward, np.ndarray):
                    reward = reward[0]
                if isinstance(done, np.ndarray):
                    done = done[0]
                if isinstance(info, list):
                    info = info[0]
                
                episode_reward += reward
                
                if 'peak_displacement' in info:
                    peak_disp = max(peak_disp, abs(info['roof_displacement']))
            
            rewards.append(episode_reward)
            peaks.append(peak_disp * 100)
        
        return np.mean(rewards), np.mean(peaks)
    
    
    def plot_validation_history(self, save_path: str = None):
        """Plot validation history"""
        if len(self.validation_history) == 0:
            print("No validation history to plot")
            return
        
        timesteps = [v['timestep'] for v in self.validation_history]
        peaks = [v['peak_disp'] for v in self.validation_history]
        improved = [v['improved'] for v in self.validation_history]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot all validations
        ax.plot(timesteps, peaks, 'o-', alpha=0.6, label='Validation Performance')
        
        # Highlight improvements
        improvement_steps = [t for t, imp in zip(timesteps, improved) if imp]
        improvement_peaks = [p for p, imp in zip(peaks, improved) if imp]
        ax.scatter(improvement_steps, improvement_peaks, 
                  color='green', s=100, zorder=5, label='Improvement')
        
        # Best line
        ax.axhline(y=self.best_peak_disp, color='red', linestyle='--', 
                  label=f'Best: {self.best_peak_disp:.2f} cm')
        
        ax.set_xlabel('Training Steps', fontsize=12)
        ax.set_ylabel('Validation Peak Displacement (cm)', fontsize=12)
        ax.set_title('Validation Performance During Training', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Validation history saved to {save_path}")
        
        plt.show()


# ================================================================
# CURRICULUM TRAINING FUNCTION
# ================================================================

def train_perfect_rl(
    earthquake_files,
    config=None
):
    """
    Train RL with all 4 improvements
    """
    
    if config is None:
        config = PerfectRLConfig()
    
    # Create directories
    os.makedirs(config.model_save_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.tensorboard_log, exist_ok=True)
    
    print("="*70)
    print("  PERFECT RL TRAINING - ALL 4 FIXES APPLIED")
    print("="*70)
    print("\n🎯 Improvements:")
    print("   1. ✅ Early Stopping with Validation")
    print("   2. ✅ Multi-Objective Reward Function")
    print("   3. ✅ Curriculum Learning (Progressive Force Limits)")
    print("   4. ✅ Regularization (Smoothness + Acceleration)")
    
    print(f"\n📚 Curriculum Plan:")
    total_steps = sum(stage['timesteps'] for stage in config.curriculum_stages)
    for i, stage in enumerate(config.curriculum_stages, 1):
        print(f"   Stage {i}: {stage['force_limit']/1000:.0f} kN - "
              f"{stage['timesteps']:,} steps - {stage['name']}")
    print(f"\n   Total: {total_steps:,} steps (~1.5-2 hours)")
    
    # Start training
    start_time = datetime.now()
    model = None
    early_stop_callback = None
    
    for stage_idx, stage in enumerate(config.curriculum_stages):
        stage_num = stage_idx + 1
        force_limit = stage['force_limit']
        timesteps = stage['timesteps']
        
        print(f"\n{'='*70}")
        print(f"  STAGE {stage_num}/{len(config.curriculum_stages)}: "
              f"{force_limit/1000:.0f} kN - {timesteps:,} steps")
        print(f"{'='*70}\n")
        
        # Create training environments with current force limit
        def make_env(eq_file, rank, force_lim):
            def _init():
                env = make_improved_tmd_env(eq_file, max_force=force_lim)
                env = Monitor(env)
                env.reset(seed=rank)
                return env
            return _init
        
        n_envs = min(4, len(earthquake_files))
        env_makers = []
        for i in range(n_envs):
            eq_file = earthquake_files[i % len(earthquake_files)]
            env_makers.append(make_env(eq_file, i, force_limit))
        
        if n_envs == 1:
            train_env = DummyVecEnv(env_makers)
        else:
            train_env = SubprocVecEnv(env_makers)
        
        # Validation environment
        val_env_single = make_improved_tmd_env(earthquake_files[0], max_force=force_limit)
        val_env_single = Monitor(val_env_single)
        val_env = DummyVecEnv([lambda: val_env_single])
        
        # Create or update model
        if model is None:
            # First stage - create new model
            print(f"🤖 Creating new SAC model...")
            model = SAC(
                "MlpPolicy",
                train_env,
                learning_rate=config.learning_rate,
                buffer_size=config.buffer_size,
                batch_size=config.batch_size,
                gamma=config.gamma,
                tau=config.tau,
                ent_coef=config.ent_coef,
                policy_kwargs=dict(net_arch=config.policy_layers),
                verbose=1,
                tensorboard_log=config.tensorboard_log,
                device='auto'
            )
        else:
            # Subsequent stages - update environment
            print(f"🔄 Updating model with new force limit...")
            model.set_env(train_env)
        
        # Create early stopping callback
        early_stop_callback = EarlyStoppingCallback(
            validation_env=val_env,
            validation_freq=config.validation_freq,
            patience=config.patience,
            min_improvement=config.min_improvement,
            verbose=1
        )
        
        # Train this stage
        print(f"🚀 Training stage {stage_num}...")
        
        try:
            model.learn(
                total_timesteps=timesteps,
                callback=early_stop_callback,
                reset_num_timesteps=False,  # Continue from previous
                tb_log_name=f"PerfectRL_Stage{stage_num}"
            )
            
            print(f"\n✅ Stage {stage_num} complete!")
            
        except Exception as e:
            if "early stopping" in str(e).lower():
                print(f"\n✅ Stage {stage_num} stopped early (optimal found)")
            else:
                print(f"\n⚠️  Stage {stage_num} error: {e}")
                break
        
        # Save stage checkpoint
        stage_path = os.path.join(config.model_save_dir, 
                                  f"stage{stage_num}_final.zip")
        model.save(stage_path)
        print(f"💾 Stage checkpoint saved: {stage_path}")
    
    # Training complete
    training_time = datetime.now() - start_time
    print(f"\n{'='*70}")
    print(f"  🎉 TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"\n   Total time: {training_time}")
    print(f"   Best validation: {early_stop_callback.best_peak_disp:.2f} cm")
    
    # Save final model
    final_path = os.path.join(config.model_save_dir, "perfect_rl_final.zip")
    model.save(final_path)
    print(f"   Final model: {final_path}")
    
    # Plot validation history
    plot_path = os.path.join(config.log_dir, "validation_history.png")
    early_stop_callback.plot_validation_history(save_path=plot_path)
    
    return model, early_stop_callback


# ================================================================
# COMPARISON FUNCTION
# ================================================================

def compare_models(
    baseline_model_path: str,
    perfect_model_path: str,
    earthquake_file: str
):
    """
    Compare baseline vs perfect RL models
    """
    
    print("\n" + "="*70)
    print("  BASELINE vs PERFECT RL COMPARISON")
    print("="*70 + "\n")
    
    # Load models
    print("Loading models...")
    baseline_model = SAC.load(baseline_model_path)
    perfect_model = SAC.load(perfect_model_path)
    
    # Create test environment
    test_env_base = make_improved_tmd_env(earthquake_file, max_force=150000)
    test_env_perf = make_improved_tmd_env(earthquake_file, max_force=150000)
    
    results = {}
    
    # Test baseline
    print("\n1️⃣ Testing BASELINE model (150 kN, 500k)...")
    obs, _ = test_env_base.reset()
    done = False
    truncated = False
    baseline_peak = 0
    baseline_forces = []
    
    while not (done or truncated):
        action, _ = baseline_model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = test_env_base.step(action)
        baseline_peak = max(baseline_peak, abs(info['roof_displacement']))
        baseline_forces.append(abs(info['control_force']) / 1000)
    
    results['baseline'] = {
        'peak_cm': baseline_peak * 100,
        'mean_force_kN': np.mean(baseline_forces),
        'max_force_kN': np.max(baseline_forces)
    }
    
    print(f"   Peak: {results['baseline']['peak_cm']:.2f} cm")
    print(f"   Mean force: {results['baseline']['mean_force_kN']:.2f} kN")
    
    # Test perfect
    print("\n2️⃣ Testing PERFECT model (all 4 fixes)...")
    obs, _ = test_env_perf.reset()
    done = False
    truncated = False
    perfect_peak = 0
    perfect_forces = []
    
    while not (done or truncated):
        action, _ = perfect_model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = test_env_perf.step(action)
        perfect_peak = max(perfect_peak, abs(info['roof_displacement']))
        perfect_forces.append(abs(info['control_force']) / 1000)
    
    results['perfect'] = {
        'peak_cm': perfect_peak * 100,
        'mean_force_kN': np.mean(perfect_forces),
        'max_force_kN': np.max(perfect_forces)
    }
    
    print(f"   Peak: {results['perfect']['peak_cm']:.2f} cm")
    print(f"   Mean force: {results['perfect']['mean_force_kN']:.2f} kN")
    
    # Comparison
    improvement = (results['baseline']['peak_cm'] - results['perfect']['peak_cm']) / results['baseline']['peak_cm'] * 100
    
    print(f"\n{'='*70}")
    print(f"  RESULTS")
    print(f"{'='*70}")
    print(f"\n   Baseline:  {results['baseline']['peak_cm']:.2f} cm")
    print(f"   Perfect:   {results['perfect']['peak_cm']:.2f} cm")
    print(f"   Improvement: {improvement:+.1f}%")
    
    if improvement > 5:
        print(f"\n   🎉 PERFECT model is {improvement:.1f}% BETTER!")
    elif improvement > 0:
        print(f"\n   ✅ PERFECT model is slightly better ({improvement:.1f}%)")
    else:
        print(f"\n   ⚠️  Baseline was better (perfect is {-improvement:.1f}% worse)")
    
    # vs Fuzzy
    fuzzy_peak = 26.0  # Your fuzzy result
    vs_fuzzy = (fuzzy_peak - results['perfect']['peak_cm']) / fuzzy_peak * 100
    
    print(f"\n   vs Fuzzy (26.0 cm): {vs_fuzzy:+.1f}%")
    
    if results['perfect']['peak_cm'] < fuzzy_peak:
        print(f"   🏆 BEATS FUZZY by {vs_fuzzy:.1f}%!")
    else:
        gap = results['perfect']['peak_cm'] - fuzzy_peak
        print(f"   Still behind fuzzy by {gap:.2f} cm")
    
    print(f"{'='*70}\n")
    
    return results


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Perfect RL with all 4 fixes')
    parser.add_argument('--earthquakes', nargs='+', required=True,
                       help='Earthquake files for training')
    parser.add_argument('--compare', type=str,
                       help='Path to baseline model for comparison')
    parser.add_argument('--test-earthquake', type=str,
                       help='Earthquake file for testing comparison')
    
    args = parser.parse_args()
    
    # Train
    print("\n🚀 Starting Perfect RL Training with ALL 4 FIXES...\n")
    model, callback = train_perfect_rl(args.earthquakes)
    
    # Compare if requested
    if args.compare and args.test_earthquake:
        compare_models(
            args.compare,
            os.path.join(PerfectRLConfig.model_save_dir, "perfect_rl_final.zip"),
            args.test_earthquake
        )
    
    print("\n✅ All done! Check perfect_rl_models/ for trained models.")
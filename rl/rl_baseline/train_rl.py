"""
REINFORCEMENT LEARNING TRAINING - TMD CONTROLLER
================================================

Train a Soft Actor-Critic (SAC) agent to control a TMD system

This learns optimal control from scratch by practicing on thousands of earthquakes

Author: Siddharth
Date: December 2025
"""

import numpy as np
import torch
import os
from datetime import datetime
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
from tmd_environment import TMDBuildingEnv, make_tmd_env


# ================================================================
# CONFIGURATION
# ================================================================

class TrainingConfig:
    """Training hyperparameters"""
    
    # Training
    total_timesteps = 500_000  # Total training steps (increase to 1M for better results)
    learning_rate = 3e-4
    batch_size = 256
    buffer_size = 100_000
    
    # Network architecture
    policy_layers = [256, 256]  # Hidden layers for actor/critic
    
    # SAC specific
    gamma = 0.99  # Discount factor
    tau = 0.005   # Soft update coefficient
    ent_coef = 'auto'  # Automatic entropy tuning
    
    # Logging & saving
    log_interval = 10  # Episodes between logs
    save_freq = 10_000  # Steps between checkpoints
    eval_freq = 10_000  # Steps between evaluations
    n_eval_episodes = 5  # Episodes per evaluation
    
    # Paths
    model_save_dir = "rl_models"
    log_dir = "rl_logs"
    tensorboard_log = "rl_tensorboard"


# ================================================================
# CUSTOM CALLBACK FOR LOGGING
# ================================================================

class TMDTrainingCallback(BaseCallback):
    """
    Custom callback for logging TMD-specific metrics
    """
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.peak_displacements = []
        
    def _on_step(self) -> bool:
        # Check if episode finished
        if self.locals['dones'][0]:
            # Extract info from last step
            info = self.locals['infos'][0]
            #infos = self.locals.get("infos", [])
            
            # Log episode stats
            self.episode_rewards.append(info["episode"]["r"])
            self.episode_lengths.append(info["episode"]["l"])
            
            if 'peak_displacement' in info:
                peak_disp_cm = info['peak_displacement'] * 100
                self.peak_displacements.append(peak_disp_cm)
                
                # Log to console periodically
                print(f"\nEpisode {len(self.episode_rewards)} completed:")
                print(f"  Reward: {info['episode']['r']:.2f}")
                print(f"  Length: {info['episode']['l']} timesteps")
                print(f"  Peak Displacement: {peak_disp_cm:.2f} cm")
                if len(self.episode_rewards) % 10 == 0:
                    avg_reward = np.mean(self.episode_rewards[-10:])
                    avg_peak = np.mean(self.peak_displacements[-10:])
                    
                    print(f"\nEpisode {len(self.episode_rewards)}:")
                    print(f"  Avg reward (last 10): {avg_reward:.2f}")
                    print(f"  Avg peak disp (last 10): {avg_peak:.2f} cm")
                    print(f"  Total timesteps: {self.num_timesteps}")
        
        return True
    
    def plot_training_progress(self, save_path=None):
        """Plot training progress"""
        if len(self.episode_rewards) == 0:
            print("No episodes completed yet")
            return
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot rewards
        axes[0].plot(self.episode_rewards, alpha=0.3, label='Episode reward')
        # Moving average
        window = min(50, len(self.episode_rewards) // 10)
        if window > 1:
            ma = np.convolve(self.episode_rewards, np.ones(window)/window, mode='valid')
            axes[0].plot(range(window-1, len(self.episode_rewards)), ma, 
                        label=f'{window}-episode MA', linewidth=2)
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Total Reward')
        axes[0].set_title('Training Rewards')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot peak displacements
        if len(self.peak_displacements) > 0:
            axes[1].plot(self.peak_displacements, alpha=0.3, label='Peak displacement')
            if window > 1:
                ma = np.convolve(self.peak_displacements, np.ones(window)/window, mode='valid')
                axes[1].plot(range(window-1, len(self.peak_displacements)), ma, 
                            label=f'{window}-episode MA', linewidth=2)
            axes[1].set_xlabel('Episode')
            axes[1].set_ylabel('Peak Displacement (cm)')
            axes[1].set_title('Peak Roof Displacement per Episode')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Training progress saved to {save_path}")
        
        plt.show()


# ================================================================
# TRAINING ENVIRONMENT CREATION
# ================================================================

def create_training_envs(earthquake_files, n_envs=4):
    """
    Create multiple parallel environments for faster training
    
    Args:
        earthquake_files: List of paths to earthquake CSV files
        n_envs: Number of parallel environments
        
    Returns:
        Vectorized environment
    """
    
    def make_env(eq_file, rank):
        def _init():
            env = make_tmd_env(eq_file)
            env = Monitor(env)
            env.reset(seed=rank)
            #env.seed(rank)
            return env
        return _init
    
    # Create list of environment makers
    # Cycle through earthquake files if we have more envs than files
    env_makers = []
    for i in range(n_envs):
        eq_file = earthquake_files[i % len(earthquake_files)]
        env_makers.append(make_env(eq_file, i))
    
    # Create vectorized environment
    if n_envs == 1:
        env = DummyVecEnv(env_makers)
    else:
        env = SubprocVecEnv(env_makers)
    
    return env


# ================================================================
# MAIN TRAINING FUNCTION
# ================================================================

def train_rl_controller(
    earthquake_files,
    config=None,
    resume_from=None
):
    """
    Train SAC agent for TMD control
    
    Args:
        earthquake_files: List of earthquake CSV files for training
        config: TrainingConfig instance (uses default if None)
        resume_from: Path to checkpoint to resume from (optional)
        
    Returns:
        Trained model
    """
    
    if config is None:
        config = TrainingConfig()
    
    # Create directories
    os.makedirs(config.model_save_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.tensorboard_log, exist_ok=True)
    
    print("="*70)
    print("  REINFORCEMENT LEARNING TRAINING - TMD CONTROLLER")
    print("="*70)
    print(f"\n📋 Training Configuration:")
    print(f"   Total timesteps: {config.total_timesteps:,}")
    print(f"   Learning rate: {config.learning_rate}")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Network: {config.policy_layers}")
    print(f"   Earthquake files: {len(earthquake_files)}")
    
    # Create training environment
    print(f"\n🏗️  Creating training environments...")
    n_parallel = min(4, len(earthquake_files))  # Up to 4 parallel envs
    train_env = create_training_envs(earthquake_files, n_envs=n_parallel)
    
    # Create evaluation environment (use first earthquake)
    print(f"🏗️  Creating evaluation environment...")
    eval_env = DummyVecEnv([lambda: Monitor(make_tmd_env(earthquake_files[0]))])
    
    # Create or load model
    if resume_from and os.path.exists(resume_from):
        print(f"\n📂 Loading model from {resume_from}...")
        model = SAC.load(resume_from, env=train_env)
        print(f"   ✅ Model loaded, continuing training")
    else:
        print(f"\n🤖 Creating new SAC model...")
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
            device='auto'  # Automatically use GPU if available
        )
        print(f"   ✅ Model created")
        print(f"   Device: {model.device}")
    
    # Create callbacks
    training_callback = TMDTrainingCallback()
    
    checkpoint_callback = CheckpointCallback(
        save_freq=config.save_freq,
        save_path=config.model_save_dir,
        name_prefix='tmd_sac_checkpoint'
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=config.model_save_dir,
        log_path=config.log_dir,
        eval_freq=config.eval_freq,
        n_eval_episodes=config.n_eval_episodes,
        deterministic=True,
        render=False
    )
    
    # Train
    print(f"\n🚀 Starting training...")
    print(f"   This will take 12-24 hours on CPU, 2-4 hours on GPU")
    print(f"   You can monitor progress in TensorBoard:")
    print(f"   tensorboard --logdir {config.tensorboard_log}\n")
    
    start_time = datetime.now()
    
    try:
        model.learn(
            total_timesteps=config.total_timesteps,
            callback=[training_callback, checkpoint_callback, eval_callback],
            log_interval=config.log_interval,
            tb_log_name="SAC_TMD"
        )
        
        training_time = datetime.now() - start_time
        print(f"\n✅ Training complete!")
        print(f"   Training time: {training_time}")
        
        # Save final model
        final_model_path = os.path.join(config.model_save_dir, "tmd_sac_final.zip")
        model.save(final_model_path)
        print(f"   Final model saved: {final_model_path}")
        
        # Plot training progress
        training_callback.plot_training_progress(
            save_path=os.path.join(config.log_dir, "training_progress.png")
        )
        
        return model, training_callback
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Training interrupted by user")
        interrupted_path = os.path.join(config.model_save_dir, "tmd_sac_interrupted.zip")
        model.save(interrupted_path)
        print(f"   Progress saved: {interrupted_path}")
        return model, training_callback


# ================================================================
# QUICK TRAINING SCRIPT
# ================================================================

def quick_train_example():
    """
    Quick training example with synthetic earthquakes
    
    Use this for testing before running full training
    """
    
    print("\n⚡ QUICK TRAINING MODE - Creating synthetic earthquakes...")
    
    # Create synthetic earthquake data
    os.makedirs("temp_earthquakes", exist_ok=True)
    
    earthquake_files = []
    for i in range(3):
        # Generate synthetic earthquake
        t = np.linspace(0, 20, 1000)
        freq = 0.8 + 0.4 * np.random.random()  # Random frequency
        mag = 2.0 + 2.0 * np.random.random()   # Random magnitude
        
        accel = mag * np.sin(2 * np.pi * freq * t) * np.exp(-0.1 * t)
        accel += 0.5 * np.random.randn(len(t))  # Add noise
        
        # Save to file
        filename = f"temp_earthquakes/synthetic_eq_{i+1}.csv"
        np.savetxt(filename, np.column_stack([t, accel]), 
                   delimiter=',', header='time,acceleration', comments='')
        earthquake_files.append(filename)
    
    print(f"   Created {len(earthquake_files)} synthetic earthquakes")
    
    # Quick training config
    config = TrainingConfig()
    config.total_timesteps = 10_000  # Much shorter for testing
    config.save_freq = 2_000
    config.eval_freq = 2_000
    
    # Train
    model, callback = train_rl_controller(earthquake_files, config=config)
    
    return model, callback


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train RL controller for TMD')
    parser.add_argument('--quick', action='store_true', 
                       help='Quick training mode with synthetic earthquakes')
    parser.add_argument('--earthquakes', nargs='+', 
                       help='Paths to earthquake CSV files')
    parser.add_argument('--timesteps', type=int, default=500_000,
                       help='Total training timesteps')
    parser.add_argument('--resume', type=str, 
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    if args.quick:
        # Quick test mode
        model, callback = quick_train_example()
    else:
        # Full training mode
        if args.earthquakes is None:
            print("❌ Error: Please provide earthquake files with --earthquakes")
            print("\nExample:")
            print("  python train_rl.py --earthquakes datasets/TEST3*.csv")
            exit(1)
        
        config = TrainingConfig()
        config.total_timesteps = args.timesteps
        
        model, callback = train_rl_controller(
            args.earthquakes,
            config=config,
            resume_from=args.resume
        )
    
    print("\n🎉 Training session complete!")
    print("   Next steps:")
    print("   1. Test the model with: python test_rl_model.py")
    print("   2. Add to your API for deployment")

"""
v11 Advanced PPO Training - With File Logging and DCR/ISDR Metrics
===================================================================

This script extends v9 with:
1. Comprehensive file logging (all console output saved to log file)
2. DCR and ISDR% metrics logged to TensorBoard
3. Enhanced TensorboardCallback for structural metrics
4. All v9 optimizations preserved

Key features from v9:
- Larger n_steps (2048-8192) for reduced variance
- Balanced batch_size (256-512) and n_epochs (10-20)
- Smoother learning rate transitions with cosine annealing
- Refined entropy scheduling
- Deeper network architecture (3 layers)
- Tighter value function clipping (0.15)

Usage: python train_v11.py

Author: Siddharth
Date: January 2026
"""

import numpy as np
import os
import glob
import random
import argparse
import logging
import sys
from datetime import datetime
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from torch.utils.tensorboard import SummaryWriter

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'restapi', 'rl_cl'))


from tmd_environment_adaptive_reward import make_improved_tmd_env
from ppo_config_v9_advanced import (
    V9AdvancedConfig,
    V9PPOHyperparameters,
    V9CurriculumStages
)


class TeeLogger:
    """
    Writes to both console and file simultaneously

    Compatible with tqdm progress bars by properly handling carriage returns
    and terminal control sequences.
    """
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'a', encoding='utf-8')
        self._last_line_was_progress = False

    def write(self, message):
        # Write to terminal (supports all terminal control codes for tqdm)
        self.terminal.write(message)

        # For file logging: filter out progress bar artifacts while keeping everything else
        if not message:
            return

        # Skip pure carriage return lines (progress bar updates)
        # But keep lines that have \r followed by actual content and \n
        if message == '\r' or (message.startswith('\r') and '\n' not in message):
            # Pure progress bar update - skip logging
            return

        # Log everything else (including lines with \n)
        # Remove leading \r if present (from progress bar line replacements)
        clean_message = message.lstrip('\r') if message.startswith('\r') else message
        self.log.write(clean_message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

    # Add properties that tqdm checks for
    @property
    def encoding(self):
        return self.terminal.encoding

    def isatty(self):
        return self.terminal.isatty()


class TensorboardCallback(BaseCallback):
    """
    Custom callback for logging additional metrics to TensorBoard

    Logs:
    - Stage information
    - Learning rate (current value from schedule)
    - Entropy coefficient
    - Episode metrics (peak displacement, DCR, ISDR, etc.)
    """

    def __init__(self, stage_num, stage_name, log_dir ,verbose=0):
        super().__init__(verbose)
        self.stage_num = stage_num
        self.stage_name = stage_name
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=log_dir)
        # Track episode-level metrics
        self.episode_isdrs = []
        self.episode_dcrs = []
        self.episode_peak_disps = []
        self.step_count = 0

    def _on_step(self) -> bool:
        """Called at every step - extract episode metrics when available"""

         # Log PPO-specific metrics from the model
        if hasattr(self.model, 'logger') and hasattr(self.model.logger, 'name_to_value'):
            for key, value in self.model.logger.name_to_value.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    self.writer.add_scalar(f'ppo/{key}', value, self.step_count)
        

        # Check if we have environment info with completed episodes
        if len(self.locals.get('infos', [])) > 0:
            for info in self.locals['infos']:
                # Episode metrics are available when episode ends
                if 'episode' in info:
                    # Extract structural metrics from episode info
                    if 'max_isdr_percent' in info:
                        self.episode_isdrs.append(info['max_isdr_percent'])
                    if 'max_dcr' in info or 'DCR' in info:
                        dcr_value = info.get('max_dcr', info.get('DCR', None))
                        if dcr_value is not None:
                            self.episode_dcrs.append(dcr_value)
                    if 'peak_displacement' in info:
                        self.episode_peak_disps.append(abs(info['peak_displacement']) * 100)  # Convert to cm
        
        self.step_count += 1                
        return True

    def _on_training_end(self) -> None:
        self.writer.close()

    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout (after collecting n_steps)"""
        # Log current learning rate (handle both callable and fixed)
        if callable(self.model.learning_rate):
            # For schedules, evaluate at current progress
            progress = 1.0 - (self.num_timesteps / self.model._total_timesteps)
            current_lr = self.model.learning_rate(progress)
        else:
            current_lr = self.model.learning_rate

        self.logger.record("train/learning_rate", current_lr)
        self.logger.record("train/entropy_coef", self.model.ent_coef)
        self.logger.record("stage/stage_number", self.stage_num)

        # Log structural safety metrics (averages over recent episodes)
        if len(self.episode_isdrs) > 0:
            avg_isdr = np.mean(self.episode_isdrs)
            max_isdr = np.max(self.episode_isdrs)
            self.logger.record("metrics/avg_isdr_percent", avg_isdr)
            self.logger.record("metrics/max_isdr_percent", max_isdr)
            self.episode_isdrs = []  # Reset for next rollout

        if len(self.episode_dcrs) > 0:
            avg_dcr = np.mean(self.episode_dcrs)
            max_dcr = np.max(self.episode_dcrs)
            self.logger.record("metrics/avg_dcr", avg_dcr)
            self.logger.record("metrics/max_dcr", max_dcr)
            self.episode_dcrs = []  # Reset for next rollout

        if len(self.episode_peak_disps) > 0:
            avg_disp = np.mean(self.episode_peak_disps)
            max_disp = np.max(self.episode_peak_disps)
            self.logger.record("metrics/avg_peak_displacement_cm", avg_disp)
            self.logger.record("metrics/max_peak_displacement_cm", max_disp)
            self.episode_peak_disps = []  # Reset for next rollout


def make_env_factory(train_files, force_limit, reward_scale=1.0):
    """Factory function to create environment makers"""
    def _make_env():
        eq_file = random.choice(train_files)
        env = make_improved_tmd_env(eq_file, max_force=force_limit, reward_scale=reward_scale)
        return Monitor(env)
    return _make_env


def create_parallel_envs(train_files, force_limit, reward_scale=1.0, n_envs=4):
    """Create parallel training environments"""
    env_fns = [make_env_factory(train_files, force_limit, reward_scale) for _ in range(n_envs)]

    try:
        env = SubprocVecEnv(env_fns)
        print(f"   ✓ Using {n_envs} parallel environments (SubprocVecEnv)\n")
    except Exception as e:
        print(f"   ⚠ SubprocVecEnv failed, falling back to DummyVecEnv")
        env = DummyVecEnv(env_fns)
        print(f"   ✓ Using {n_envs} sequential environments (DummyVecEnv - slower)\n")

    return env


def get_entropy_coefficient(stage):
    """
    Get entropy coefficient for the stage

    Args:
        stage: Stage configuration dict

    Returns:
        Entropy coefficient (float)

    Note: PPO doesn't support callable ent_coef schedules in stable-baselines3,
    so we use fixed values per stage. For annealing, use the average value.
    """
    if stage.get('ent_schedule', False):
        # Use average of initial and final for this stage
        initial_ent = stage['ent_coef']
        final_ent = stage.get('final_ent', initial_ent * 0.1)
        avg_ent = (initial_ent + final_ent) / 2.0
        return avg_ent
    else:
        return stage['ent_coef']


def create_v9_ppo_model(env, stage, device, tensorboard_log=None):
    """
    Create PPO model with v9 advanced configuration

    Args:
        env: Training environment
        stage: Stage configuration dict
        device: 'cuda' or 'cpu'
        tensorboard_log: Path to TensorBoard log directory (optional)

    Returns:
        PPO model instance
    """
    # Get learning rate schedule
    lr_schedule = V9CurriculumStages.get_learning_rate_schedule(stage)

    # Get entropy coefficient (fixed value, not callable)
    ent_coef = get_entropy_coefficient(stage)

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
        print(f"      ent_coef: {ent_coef:.4f} (averaged from {stage['ent_coef']} → {stage['final_ent']})")
    else:
        print(f"      ent_coef: {ent_coef} (fixed)")

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
        ent_coef=ent_coef,  # Fixed float value
        vf_coef=V9PPOHyperparameters.VF_COEF,
        max_grad_norm=V9PPOHyperparameters.MAX_GRAD_NORM,
        policy_kwargs=policy_kwargs,
        verbose=V9PPOHyperparameters.VERBOSE,
        tensorboard_log=tensorboard_log,  # Enable TensorBoard logging
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

    # Close old environment if it exists
    if env_old is not None:
        env_old.close()

    # Update learning rate schedule
    lr_schedule = V9CurriculumStages.get_learning_rate_schedule(stage)
    if callable(lr_schedule):
        print(f"      learning_rate: Cosine schedule ({stage['learning_rate']:.0e} → {stage['final_lr']:.0e})")
        model.learning_rate = lr_schedule
    else:
        print(f"      learning_rate: {lr_schedule:.0e} (fixed)")
        model.learning_rate = lr_schedule

    # Update entropy coefficient (fixed value, not callable)
    ent_coef = get_entropy_coefficient(stage)
    if stage.get('ent_schedule'):
        print(f"      ent_coef: {ent_coef:.4f} (averaged from {stage['ent_coef']} → {stage['final_ent']})")
    else:
        print(f"      ent_coef: {ent_coef} (fixed)")
    model.ent_coef = ent_coef

    # Recreate optimizer with new learning rate
    # Note: If using schedule, get initial value
    initial_lr = lr_schedule(1.0) if callable(lr_schedule) else lr_schedule
    model.policy.optimizer = model.policy.optimizer_class(
        model.policy.parameters(),
        lr=initial_lr,
        **model.policy.optimizer_kwargs
    )

    print()


def find_last_completed_stage(model_dir, stages):
    """
    Find the last completed stage by checking for checkpoint files

    Args:
        model_dir: Directory where checkpoints are saved
        stages: List of stage configurations

    Returns:
        Tuple of (last_completed_stage_idx, checkpoint_path) or (None, None) if no checkpoints
    """
    if not os.path.exists(model_dir):
        return None, None

    # Check stages in reverse order (latest first)
    for stage_idx in range(len(stages) - 1, -1, -1):
        stage_num = stage_idx + 1
        stage = stages[stage_idx]
        force_limit = stage['force_limit']

        # Only look for successfully completed stages (not INTERRUPTED or ERROR checkpoints)
        checkpoint_path = f"{model_dir}/stage{stage_num}_{force_limit//1000}kN.zip"

        if os.path.exists(checkpoint_path):
            # Verify it's not a partial/emergency checkpoint
            if "INTERRUPTED" not in checkpoint_path and "ERROR" not in checkpoint_path:
                return stage_idx, checkpoint_path

    return None, None


def test_on_earthquake(model, test_file, force_limit):
    """Test model on single earthquake"""
    test_env = make_improved_tmd_env(test_file, max_force=force_limit, reward_scale=1.0)
    obs, _ = test_env.reset()
    done = False
    peak_disp = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = test_env.step(action)
        peak_disp = max(peak_disp, abs(info['roof_displacement']))
        done = done or truncated

    # Get episode metrics (includes max_isdr_percent and DCR from get_episode_metrics())
    metrics = test_env.get_episode_metrics()
    metrics['peak_disp'] = peak_disp * 100  # Convert to cm

    # The metrics dict already contains max_isdr_percent and DCR from get_episode_metrics()
    # No need to extract from environment attributes or override them
    test_env.close()

    return metrics


def parse_args():
    """
    Parse command-line arguments for training configuration

    Returns:
        Namespace with train_dir, test_dir, model_dir, log_dir, and run_name
    """
    parser = argparse.ArgumentParser(
        description='Train v11 Advanced PPO with file logging and DCR/ISDR metrics',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--train-dir',
        type=str,
        default='../../matlab/datasets/training/training_set_v2',
        help='Directory containing training dataset variants'
    )

    parser.add_argument(
        '--test-dir',
        type=str,
        default='../../matlab/datasets/test',
        help='Directory containing test earthquake files  (default: ../../matlab/datasets/training/training_set_v2)'
    )

    parser.add_argument(
        '--model-dir',
        type=str,
        default='models/rl_v11_advanced',
        help='Directory to save model checkpoints and training state (default: models/rl_v11_advanced)'
    )

    parser.add_argument(
        '--log-dir',
        type=str,
        default='logs/rl_v11_advanced',
        help='Directory for TensorBoard logs (default: logs/tensorboard)'
    )

    parser.add_argument(
        '--run-name',
        type=str,
        default=None,
        help='Name for this training run (for TensorBoard organization). Default: auto-generated from timestamp'
    )

    return parser.parse_args()


def train_v11_advanced(train_dir=None, test_dir=None, model_dir=None, log_dir=None, run_name=None):
    """
    Main training function with v11 enhanced configuration

    V11 enhancements:
    - All console output logged to file
    - DCR and ISDR metrics tracked in TensorBoard
    - Enhanced monitoring for structural safety

    Args:
        train_dir: Directory containing training dataset variants (default: ../../matlab/datasets/training_set_v2)
        test_dir: Directory containing test earthquake files (default: ../../matlab/datasets)
        model_dir: Directory to save model checkpoints (default: models/rl_v11_advanced)
        log_dir: Directory for TensorBoard logs (default: logs/rl_v11_advanced)
        run_name: Optional run name for organizing experiments (auto-generated if None)
    """

    # Set default directories if not provided
    if train_dir is None:
        train_dir = '../../matlab/datasets/training/training_set_v2'
    if test_dir is None:
        test_dir = '../../matlab/datasets/test'
    if model_dir is None:
        model_dir = 'models/rl_v11_advanced'
    if log_dir is None:
        log_dir = 'logs/rl_v11_advanced'

    # Auto-generate run name from timestamp if not provided
    if run_name is None:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"run_{model_dir.split('/')[-1]}_{run_name}"

    print(f"\n{'='*70}")
    print(f" train_dir: {train_dir}")
    print(f" test_dir:  {test_dir}")    
    print(f" model_dir: {model_dir}")
    print(f" log_dir:   {log_dir}")
    print(f" run_name:  {run_name}")
    print(f"{'='*70}\n")

    # Add run_name as subdirectory
    model_dir = os.path.join(model_dir, run_name)
    log_dir = os.path.join(log_dir, run_name)

    # Create log directory first (needed for file logging)
    os.makedirs(log_dir, exist_ok=True)

    # Create output directory for model checkpoints
    os.makedirs(model_dir, exist_ok=True)

    # Create TensorBoard log directory
    tensorboard_log = log_dir
    os.makedirs(tensorboard_log, exist_ok=True)

    print(f"📁 Output configuration:")
    print(f"   Model directory: {model_dir}")
    print(f"   Log directory: {tensorboard_log}")
    print(f"   To view: tensorboard --logdir={log_dir.split('/')[0]}\n")


    # Set up file logging (REDIRECT ALL CONSOLE OUTPUT TO FILE)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(model_dir, f"training_{timestamp}.log")

    print(f"📝 Setting up file logging...")
    print(f"   Log file: {log_file}\n")

    tee_logger = TeeLogger(log_file)
    sys.stdout = tee_logger

    print(f"📝 Logging all output to: {log_file}\n")

    # Auto-detect device
    gpu_name = None
    if torch.cuda.is_available():
        device = 'cuda'
        gpu_name = torch.cuda.get_device_name(0)
        print("="*70)
        print("  V11 ADVANCED PPO TRAINING (GPU ACCELERATED)")
        print("="*70)
        print(f"\n🚀 GPU: {gpu_name}")
        print(f"   Device: {device}\n")
    else:
        device = 'cpu'
        print("="*70)
        print("  V11 ADVANCED PPO TRAINING")
        print("="*70)
        print("\n⚠️  No GPU detected - using CPU\n")

    # Print configuration
    print(f"🏷️  Run name: {run_name}")
    print(f"📁 Training directory: {train_dir}")
    print(f"📁 Test directory: {test_dir}\n")

    # Print improvements summary
    V9AdvancedConfig.print_improvements_summary()

    # Print curriculum
    V9AdvancedConfig.curriculum.print_curriculum_summary()

    # Dataset paths - use provided directories
    test_files = {
        'M4.5': os.path.join(test_dir, "PEER_small_M4.5_PGA0.25g.csv"),
        'M5.7': os.path.join(test_dir, "PEER_moderate_M5.7_PGA0.35g.csv"),
        'M7.4': os.path.join(test_dir, "PEER_high_M7.4_PGA0.75g.csv"),
        'M8.4': os.path.join(test_dir, "PEER_insane_M8.4_PGA0.9g.csv")
    }

    # Find training files
    print("📁 Locating training files...")
    train_files = {}
    for magnitude in ['M4.5', 'M5.7', 'M7.4', 'M8.4']:
        pattern = os.path.join(train_dir, f"TRAIN_{magnitude}_*.csv")
        files = sorted(glob.glob(pattern))

        if not files:
            print(f"   ❌ {magnitude}: No training files found!")
            print(f"      Expected: {pattern}")
            return None

        train_files[magnitude] = files
        print(f"   ✓ {magnitude}: {len(files)} variants")

    print()

   
    # Check for existing checkpoints
    n_envs = V9PPOHyperparameters.N_ENVS
    stages = V9CurriculumStages.STAGES

    last_completed_idx, checkpoint_path = find_last_completed_stage(model_dir, stages)

    if last_completed_idx is not None:
        print(f"\n{'='*70}")
        print(f"  🔄 RESUMING FROM CHECKPOINT")
        print(f"{'='*70}")
        print(f"\n   Found checkpoint: {checkpoint_path}")
        print(f"   Last completed: Stage {last_completed_idx + 1}/{len(stages)}")
        print(f"   Loading model from checkpoint...\n")

        # Load the checkpoint
        model = PPO.load(checkpoint_path, device=device)
        print(f"   ✅ Model loaded successfully!")
        print(f"   Will resume from Stage {last_completed_idx + 2}/{len(stages)}\n")

        start_stage_idx = last_completed_idx + 1
    else:
        print(f"\n   No checkpoints found - starting fresh training\n")
        model = None
        start_stage_idx = 0

    # Training loop
    start_time = datetime.now()

    for stage_idx, stage in enumerate(stages):
        # Skip already completed stages
        if stage_idx < start_stage_idx:
            continue
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
        print(f"   Timesteps: {timesteps:,}")

        # Get reward_scale from stage config (default to 1.0 if not specified)
        reward_scale = stage.get('reward_scale', 1.0)
        print(f"   Reward scale: {reward_scale} (fixed, no adaptive scaling)\n")

        # Create parallel environments
        env = create_parallel_envs(available_files, force_limit, reward_scale, n_envs)

        # Create or update model
        if model is None:
            # Starting fresh (no checkpoint)
            print(f"🤖 Creating v11 Advanced PPO model...")
            print(f"   Device: {device}\n")
            model = create_v9_ppo_model(env, stage, device, tensorboard_log)
        elif stage_idx == start_stage_idx and start_stage_idx > 0:
            # Resuming from checkpoint - need to update for new stage
            print(f"🔄 Resuming training from checkpoint...\n")
            update_model_for_new_stage(model, env, stage)
        else:
            # Continuing from previous stage in same run
            print(f"🔄 Continuing from Stage {stage_num-1}...\n")
            update_model_for_new_stage(model, env, stage)

         # IMPROVEMENT: Granular checkpoint saving (every 50k steps)
        checkpoint_dir = f"{model_dir}/stage{stage_num}_checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_callback = CheckpointCallback(
            save_freq=50000,  # Save every 50k steps
            save_path=checkpoint_dir,
            name_prefix=f"stage{stage_num}_ppo"
        )

        # Create TensorBoard callback for this stage
        tb_callback = TensorboardCallback(
            stage_num=stage_num,
            stage_name=stage['name'],
            log_dir=tensorboard_log
        )

        # Train with error handling
        try:
            print(f"🚀 Training Stage {stage_num}...")
            model.learn(
                total_timesteps=timesteps,
                reset_num_timesteps=False,
                progress_bar=True,
                #callback=[checkpoint_callback, tb_callback],
                callback=tb_callback,  # Add TensorBoard callback
                tb_log_name=f"stage{stage_num}"  # Unique name per stage
            )

            # Save checkpoint after successful training
            save_path = f"{model_dir}/stage{stage_num}_{force_limit//1000}kN.zip"
            model.save(save_path)
            print(f"\n💾 Saved: {save_path}")

            # Test on held-out earthquake
            test_file = test_files[magnitude]
            print(f"\n📊 Testing on HELD-OUT {magnitude} test earthquake...")
            print(f"   Test file: {os.path.basename(test_file)}")

            metrics = test_on_earthquake(model, test_file, force_limit)
            peak_cm = metrics['peak_disp']
            print(f"\n  [RESULTS] Stage {stage_num} - {magnitude}:")
            print(f"      Peak Roof Displacement: {metrics['peak_disp']:.2f} cm")
            print(f"      Max ISDR%: {metrics['max_isdr_percent']:.2f} %")
            print(f"      DCR: {metrics['DCR']:.2f}")
            print(f"      Mean force: {metrics['mean_force']/1000:.1f} kN")
            print(f"      RMS displacement: {metrics['rms_roof_displacement']*100:.2f} cm")
            
            print(f"\n✅ Stage {stage_num} complete!\n")

        except KeyboardInterrupt:
            print(f"\n\n⚠️  Training interrupted by user!")
            print(f"   Saving emergency checkpoint...")

            # Save emergency checkpoint
            emergency_path = f"{model_dir}/stage{stage_num}_INTERRUPTED_{force_limit//1000}kN.zip"
            model.save(emergency_path)
            print(f"   💾 Emergency checkpoint saved: {emergency_path}")
            print(f"\n   You can resume training by running the script again.")
            print(f"   Training will automatically resume from Stage {stage_num}.\n")

            # Close environment before exiting
            env.close()

            # Restore stdout and close log file
            sys.stdout = tee_logger.terminal
            tee_logger.close()
            print(f"\n📝 Training log saved to: {log_file}")
            raise

        except Exception as e:
            print(f"\n\n❌ ERROR during Stage {stage_num} training!")
            print(f"   Error: {e}")
            print(f"\n   Saving emergency checkpoint...")

            # Save emergency checkpoint
            emergency_path = f"{model_dir}/stage{stage_num}_ERROR_{force_limit//1000}kN.zip"
            model.save(emergency_path)
            print(f"   💾 Emergency checkpoint saved: {emergency_path}")
            print(f"\n   You can resume training by running the script again.")
            print(f"   Training will automatically resume from the last successful stage.\n")

            # Close environment before exiting
            env.close()

            # Restore stdout and close log file
            sys.stdout = tee_logger.terminal
            tee_logger.close()
            print(f"\n📝 Training log saved to: {log_file}")
            raise

        # Close environments after successful stage
        env.close()

    # Final model save
    training_time = datetime.now() - start_time
    final_path = f"{model_dir}/final_v11_advanced.zip"
    model.save(final_path)

    # Print summary
    print("="*70)
    print("  🎉 TRAINING COMPLETE!")
    print("="*70)
    print(f"\n   Total training time: {training_time}")
    if start_stage_idx > 0:
        print(f"   Resumed from: Stage {start_stage_idx + 1}/{len(stages)}")
        print(f"   Stages trained in this run: {len(stages) - start_stage_idx}/{len(stages)}")
    print(f"   Training device: {device.upper()}")
    if gpu_name:
        print(f"   GPU: {gpu_name}")
    print(f"   Final model: {final_path}")
    print(f"\n   V9 Advanced Features:")
    print(f"   • Deeper network: [256, 256, 256] with Tanh")
    print(f"   • Larger n_steps: 2048-8192 (reduced variance)")
    print(f"   • Balanced batch_size: 256-512")
    print(f"   • Optimized n_epochs: 10-20")
    print(f"   • Cosine LR annealing (smoother decay)")
    print(f"   • Entropy coefficient averaging")
    print(f"   • Tighter value clipping: 0.15")
    print(f"   • Automatic checkpoint recovery")
    print(f"   • TensorBoard logging enabled")
    print(f"   • Total timesteps: {V9CurriculumStages.get_total_timesteps():,}")

    print(f"\n📊 View training metrics:")
    print(f"   tensorboard --logdir={tensorboard_log}")

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
        metrics = test_on_earthquake(model, test_file, 150000)
        peak_cm = metrics['peak_disp']

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
        print(f"      v11-Advanced: {peak_cm:.2f} cm")
        print(f"      v8-Baseline: {v8_peak:.2f} cm")
        print(f"      Δ from v8: {delta_v8:+.2f} cm {v8_status}")
        print(f"      Uncontrolled: {uncont:.2f} cm")
        print(f"      Improvement: {improvement:+.1f}% {status}")
        print(f"      ISDR: {metrics['max_isdr_percent']:.2f}%")
        print(f"      DCR: {metrics['DCR']:.2f}\n")

    print("="*70 + "\n")

    # Restore stdout and close log file
    sys.stdout = tee_logger.terminal
    tee_logger.close()
    print(f"\n📝 Training log saved to: {log_file}")

    return model


if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()

    print("\n" + "="*70)
    print(f" args: {args} ")
    print("="*70 + "\n")

    #return_code = 0

    print("\n🚀 Starting v11 Advanced PPO Training...\n")
    print("v11 Enhancements:")
    print("  ✓ All console output logged to file")
    print("  ✓ DCR and ISDR metrics in TensorBoard")
    print("  ✓ Enhanced structural safety monitoring")
    print("  ✓ Flexible dataset directories\n")

    # Train with specified directories
    model = train_v11_advanced(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        model_dir=args.model_dir,
        log_dir=args.log_dir,
        run_name=args.run_name
    )

    if model is not None:
        print("✅ v11 Advanced training complete!")
        print("\nKey improvements (v9 + v11):")
        print("  ✓ Larger n_steps (2× increase)")
        print("  ✓ Balanced batch_size and n_epochs")
        print("  ✓ Cosine LR annealing")
        print("  ✓ Deeper network architecture")
        print("  ✓ File logging with TeeLogger")
        print("  ✓ DCR/ISDR metrics tracked")
        print("  ✓ Refined value function clipping")
        print("  ✓ TensorBoard logging")
        print("  ✓ Automatic checkpoint recovery")
        print("\nExpected benefits:")
        print("  • Reduced variance in advantage estimates")
        print("  • Smoother convergence")
        print("  • Better generalization on M7.4/M8.4")
    else:
        print("❌ Training failed - check training files.")

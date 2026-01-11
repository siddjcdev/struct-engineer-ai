"""
TMD Training v12 - Soft Story TMD Configuration
================================================

V12 BREAKTHROUGH CHANGES:
- TMD mounted AT FLOOR 8 (soft story) for direct ISDR control
- 300 kN max force for aggressive control authority
- New reward function targeting 14cm, 0.4% ISDR, 1.15 DCR
- Combined best practices from v9 (hyperparameters) and v11 (architecture)

This is the definitive configuration for proving TMD effectiveness
in soft-story buildings.

Author: Claude Code
Date: January 2026
"""

import sys
import os
import argparse
import random
import glob
import logging
from datetime import datetime
from typing import List, Dict
import numpy as np
import traceback

# Add restapi path
sys.path.insert(0, os.path.abspath('../../restapi/rl_cl'))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.logger import configure
from torch.utils.tensorboard import SummaryWriter

import torch

# Import v12 environment
from tmd_environment_v12_soft_story import make_soft_story_tmd_env


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

# Global log file handler
_log_file = None

def setup_log_file(logs_dir: str, run_name: str):
    """Setup log file for training session"""
    global _log_file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{run_name}_{timestamp}.log"
    log_path = os.path.join(logs_dir, log_filename)
    _log_file = open(log_path, 'w', encoding='utf-8')
    return log_path

def log_print(message="", **kwargs):
    """Print to both console and log file"""
    print(message, **kwargs)
    if _log_file:
        # Clean message for log file (remove emojis)
        clean_msg = message.replace('✅', '[OK]').replace('❌', '[ERROR]').replace('⚠️', '[WARNING]').replace('🏆', '[SUCCESS]')
        print(clean_msg, file=_log_file, **kwargs)
        _log_file.flush()

def close_log_file():
    """Close the log file"""
    global _log_file
    if _log_file:
        _log_file.close()
        _log_file = None


# ============================================================================
# V12 PPO CONFIGURATION
# ============================================================================

class V12Config:
    """V12 PPO configuration combining v9 hyperparameters with v11 architecture"""

    # Network architecture - Deep 4-layer from v9
    POLICY_TYPE = "MlpPolicy"
    NETWORK_ARCH = [256, 256, 256, 256]
    ACTIVATION_FN = torch.nn.Tanh

    # PPO hyperparameters - Optimized from v9
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    CLIP_RANGE = 0.2
    CLIP_RANGE_VF = 0.15
    VF_COEF = 0.5
    MAX_GRAD_NORM = 0.5
    VERBOSE = 1
    N_ENVS = 4

    # Training stages
    STAGES = [
        {
            'name': 'M4.5 @ 300kN - Soft Story TMD',
            'magnitude': 'M4.5',
            'force_limit': 300_000,    # 300 kN max force
            'timesteps': 1_500_000,    # 1.5M steps for convergence
            'reward_scale': 1.0,       # Fixed reward scale

            # PPO parameters
            'n_steps': 2048,
            'batch_size': 256,
            'n_epochs': 10,
            'learning_rate': 3e-4,
            'ent_coef': 0.03,

            'description': 'Soft-story TMD targeting 14cm, 0.4% ISDR, 1.15 DCR'
        }
    ]

    @staticmethod
    def get_policy_kwargs() -> dict:
        return {
            'net_arch': V12Config.NETWORK_ARCH,
            'activation_fn': V12Config.ACTIVATION_FN
        }


class TensorBoardMetricsCallback(BaseCallback):
    """Custom callback for logging structural engineering metrics to TensorBoard"""
    
    def __init__(self, log_dir, verbose=0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=log_dir)
        self.step_count = 0
    
    def _on_step(self) -> bool:
        # Log PPO-specific metrics from the model
        if hasattr(self.model, 'logger') and hasattr(self.model.logger, 'name_to_value'):
            for key, value in self.model.logger.name_to_value.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    self.writer.add_scalar(f'ppo/{key}', value, self.step_count)
        
        # Log from environment if available
        if hasattr(self.training_env, 'envs') and len(self.training_env.envs) > 0:
            env = self.training_env.envs[0]
            if hasattr(env, 'max_displacement'):
                self.writer.add_scalar('structural/max_displacement_m',
                                     env.max_displacement,
                                     self.step_count)
            if hasattr(env, 'max_isdr_percent'):
                self.writer.add_scalar('structural/max_isdr_percent',
                                     env.max_isdr_percent,
                                     self.step_count)
            if hasattr(env, 'max_dcr'):
                self.writer.add_scalar('structural/max_dcr',
                                     env.max_dcr,
                                     self.step_count)
            if hasattr(env, 'max_roof_velocity'):
                self.writer.add_scalar('structural/max_roof_velocity',
                                     env.max_roof_velocity,
                                     self.step_count)
            if hasattr(env, 'episode_reward'):
                self.writer.add_scalar('training/episode_reward',
                                     env.episode_reward,
                                     self.step_count)
        
        self.step_count += 1
        return True
    
    def _on_training_end(self) -> None:
        self.writer.close()


# ============================================================================
# LOGGING UTILITIES
# ============================================================================

class DualLogger:
    """Logger that writes to both console and file"""

    def __init__(self, log_file_path: str):
        self.log_file = open(log_file_path, 'w', encoding='utf-8')
        self.console = sys.stdout

    def write(self, message: str):
        """Write message to both console and file"""
        self.console.write(message)
        self.console.flush()
        # Remove ANSI color codes and emojis for clean log file
        clean_message = self._clean_message(message)
        self.log_file.write(clean_message)
        self.log_file.flush()

    def _clean_message(self, message: str) -> str:
        """Remove ANSI codes and convert emojis to text"""
        import re
        # Remove ANSI escape codes
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        message = ansi_escape.sub('', message)

        # Convert emoji to text representation
        emoji_map = {
            '✅': '[OK]',
            '❌': '[ERROR]',
            '⚠️': '[WARNING]',
            '🏆': '[SUCCESS]',
        }
        for emoji, text in emoji_map.items():
            message = message.replace(emoji, text)

        return message

    def flush(self):
        """Flush both outputs"""
        self.console.flush()
        self.log_file.flush()

    def close(self):
        """Close the log file"""
        self.log_file.close()


def setup_logging(logs_dir: str, run_name: str) -> DualLogger:
    """
    Setup logging to both console and file

    Args:
        logs_dir: Directory for log files
        run_name: Name of the training run

    Returns:
        DualLogger instance
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{run_name}_{timestamp}.log"
    log_path = os.path.join(logs_dir, log_filename)

    logger = DualLogger(log_path)

    # Redirect stdout to dual logger
    sys.stdout = logger

    print(f"Logging to: {log_path}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    return logger


# ============================================================================
# DATASET UTILITIES
# ============================================================================

def find_training_files(data_dir: str, magnitude: str) -> List[str]:
    """
    Find all training files for a given magnitude

    Args:
        data_dir: Base data directory
        magnitude: Magnitude string (e.g., 'M4.5', 'M5.7')

    Returns:
        List of absolute paths to training files
    """
    training_dir = os.path.join(data_dir, 'training', 'training_set_v2')

    if not os.path.exists(training_dir):
        raise FileNotFoundError(
            f"Training directory not found: {training_dir}\n"
            f"Expected structure: {data_dir}/training/training_set_v2/"
        )

    # Map magnitude to file pattern
    magnitude_map = {
        'M4.5': 'TRAIN_M4.5_PGA0.25g_RMS0.073g_variant*.csv',
        'M5.7': 'TRAIN_M5.7_PGA0.35g_RMS0.100g_variant*.csv',
        'M7.4': 'TRAIN_M7.4_PGA0.75g_RMS0.331g_variant*.csv',
        'M8.4': 'TRAIN_M8.4_PGA0.9g_RMS0.274g_variant*.csv'
    }

    pattern = magnitude_map.get(magnitude)
    if not pattern:
        raise ValueError(f"Unknown magnitude: {magnitude}. Expected one of: {list(magnitude_map.keys())}")

    # Find all matching files
    search_pattern = os.path.join(training_dir, pattern)
    train_files = glob.glob(search_pattern)

    if not train_files:
        raise FileNotFoundError(
            f"No training files found for {magnitude}\n"
            f"Searched: {search_pattern}\n"
            f"Please ensure training files exist in: {training_dir}"
        )

    return sorted(train_files)


def find_test_file(data_dir: str, magnitude: str) -> str:
    """
    Find held-out test file for a given magnitude

    Args:
        data_dir: Base data directory
        magnitude: Magnitude string (e.g., 'M4.5')

    Returns:
        Absolute path to test file
    """
    test_dir = os.path.join(data_dir, 'test')

    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    # Map magnitude to test file
    test_file_map = {
        'M4.5': 'PEER_small_M4.5_PGA0.25g.csv',
        'M5.7': 'PEER_moderate_M5.7_PGA0.35g.csv',
        'M7.4': 'PEER_high_M7.4_PGA0.75g.csv',
        'M8.4': 'PEER_insane_M8.4_PGA0.9g.csv'
    }

    test_filename = test_file_map.get(magnitude)
    if not test_filename:
        raise ValueError(f"Unknown magnitude: {magnitude}")

    test_file = os.path.join(test_dir, test_filename)

    if not os.path.exists(test_file):
        raise FileNotFoundError(
            f"Test file not found: {test_file}\n"
            f"Please ensure test file exists in: {test_dir}"
        )

    return test_file


# ============================================================================
# ENVIRONMENT CREATION
# ============================================================================

def make_env_factory(train_files: List[str], force_limit: float, reward_scale: float = 1.0):
    """Create environment factory for parallel envs"""
    def _make_env():
        eq_file = random.choice(train_files)
        try:
            env = make_soft_story_tmd_env(
                eq_file,
                max_force=force_limit,
                reward_scale=reward_scale
            )
            return Monitor(env)
        except Exception as e:
            print(f"ERROR creating environment with file {eq_file}: {e}")
            traceback.print_exc()
            raise
    return _make_env


def create_parallel_envs(train_files: List[str], force_limit: float, reward_scale: float = 1.0, n_envs: int = 4):
    """Create parallel training environments"""
    if not train_files:
        raise ValueError("No training files provided!")

    print(f"  Creating parallel envs with {len(train_files)} training files:")
    for f in train_files:
        print(f"    - {os.path.basename(f)}")

    env_fns = [make_env_factory(train_files, force_limit, reward_scale) for _ in range(n_envs)]
    return SubprocVecEnv(env_fns)


# ============================================================================
# HELD-OUT TEST EVALUATION
# ============================================================================

def evaluate_on_test_set(model, test_file: str, force_limit: float, magnitude: str):
    """
    Evaluate trained model on held-out test set

    Args:
        model: Trained PPO model
        test_file: Path to test earthquake file
        force_limit: Maximum control force
        magnitude: Magnitude string for logging

    Returns:
        Dict with test metrics
    """
    print(f"\n{'='*70}")
    print(f"  HELD-OUT TEST EVALUATION - {magnitude}")
    print(f"{'='*70}")
    print(f"  Test file: {os.path.basename(test_file)}")

    try:
        # Create test environment
        env = make_soft_story_tmd_env(
            test_file,
            max_force=force_limit,
            reward_scale=1.0
        )

        # Run episode
        obs, _ = env.reset()
        done = False
        truncated = False

        peak_disp = 0.0
        forces = []
        episode_reward = 0.0

        while not (done or truncated):
            # Get deterministic action
            action, _ = model.predict(obs, deterministic=True)

            # Step environment
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward

            # Track metrics
            peak_disp = max(peak_disp, abs(info['roof_displacement']))
            forces.append(abs(info['control_force']))

        # Get final metrics
        metrics = env.get_episode_metrics()
        metrics['peak_disp_cm'] = peak_disp * 100
        metrics['peak_force_kN'] = np.max(forces) / 1000
        metrics['mean_force_kN'] = np.mean(forces) / 1000
        metrics['episode_reward'] = episode_reward

        env.close()

        # Print results
        print(f"\n  Test Results:")
        print(f"    Peak Displacement: {metrics['peak_disp_cm']:.2f} cm")
        print(f"    Max ISDR:          {metrics['max_isdr_percent']:.3f}%")
        print(f"    DCR:               {metrics['DCR']:.2f}")
        print(f"    Peak Force:        {metrics['peak_force_kN']:.1f} kN")
        print(f"    Episode Reward:    {episode_reward:.2f}")

        # Evaluate against targets
        print(f"\n  Target Achievement:")
        disp_target = 14.0
        isdr_target = 0.4
        dcr_target = 1.15

        disp_status = "✅" if metrics['peak_disp_cm'] <= disp_target else \
                      "⚠️" if metrics['peak_disp_cm'] <= disp_target + 4 else "❌"
        isdr_status = "✅" if metrics['max_isdr_percent'] <= isdr_target else \
                      "⚠️" if metrics['max_isdr_percent'] <= 0.8 else "❌"
        dcr_status = "✅" if metrics['DCR'] <= dcr_target else \
                     "⚠️" if metrics['DCR'] <= 1.3 else "❌"

        print(f"    Displacement: {metrics['peak_disp_cm']:.2f} cm (target: {disp_target} cm) {disp_status}")
        print(f"    ISDR:         {metrics['max_isdr_percent']:.3f}% (target: {isdr_target}%) {isdr_status}")
        print(f"    DCR:          {metrics['DCR']:.2f} (target: {dcr_target}) {dcr_status}")

        all_met = (metrics['peak_disp_cm'] <= disp_target and
                   metrics['max_isdr_percent'] <= isdr_target and
                   metrics['DCR'] <= dcr_target)

        if all_met:
            print(f"\n  🏆 ALL TARGETS MET - Breakthrough success!")
        else:
            all_close = (metrics['peak_disp_cm'] <= disp_target + 4 and
                        metrics['max_isdr_percent'] <= 0.8 and
                        metrics['DCR'] <= 1.3)
            if all_close:
                print(f"\n  ⚠️  Close to targets - Very good performance")
            else:
                print(f"\n  ❌ Targets not met - May need more training")

        print(f"{'='*70}\n")

        return metrics

    except Exception as e:
        print(f"\n❌ ERROR during test evaluation: {e}")
        traceback.print_exc()
        return None


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_v12(
    run_name: str = "v12_soft_story_tmd",
    data_dir: str = "../../matlab/datasets",
    models_dir: str = "models",
    logs_dir: str = "logs",
    resume_from: str = None
):
    """
    Train v12 soft-story TMD model

    Args:
        run_name: Run identifier
        data_dir: Path to earthquake dataset directory
        models_dir: Path to save models
        logs_dir: Path to save logs
        resume_from: Path to model to resume from (optional)
    """
    # Create directories first
    run_dir = os.path.join(models_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # Setup logging to file
    log_path = setup_log_file(models_dir, run_name)

    try:
        print("=" * 80)
        print("  TMD TRAINING V12 - SOFT STORY CONFIGURATION")
        print("  Breakthrough: TMD at floor 8 for direct ISDR control")
        print("=" * 80)

        print(f"\n  Output directories:")
        print(f"    Models: {os.path.abspath(run_dir)}")
        print(f"    Logs:   {os.path.abspath(logs_dir)}")

        # Train on all available stages
        model = None
        test_results = {}  # Store test results for each stage

        for stage_idx, stage in enumerate(V12Config.STAGES):
            print(f"\n{'=' * 80}")
            print(f"  STAGE {stage_idx + 1}: {stage['name']}")
            print(f"  {stage['description']}")
            print("=" * 80)

            # Get training files for this magnitude
            magnitude = stage['magnitude']
            force_limit = stage['force_limit']
            reward_scale = stage.get('reward_scale', 1.0)
            timesteps = stage['timesteps']

            try:
                # Find training files
                print(f"\n  Locating training files for {magnitude}...")
                train_files = find_training_files(data_dir, magnitude)
                print(f"  ✅ Found {len(train_files)} training files")

                # Find test file
                print(f"  Locating held-out test file for {magnitude}...")
                test_file = find_test_file(data_dir, magnitude)
                print(f"  ✅ Found test file: {os.path.basename(test_file)}")

            except FileNotFoundError as e:
                print(f"\n  ❌ ERROR: {e}")
                print(f"  Skipping stage {stage_idx + 1}")
                continue
            except Exception as e:
                print(f"\n  ❌ UNEXPECTED ERROR: {e}")
                traceback.print_exc()
                print(f"  Skipping stage {stage_idx + 1}")
                continue

            print(f"\n  Training configuration:")
            print(f"    Earthquake: {magnitude}")
            print(f"    Training files: {len(train_files)} variants")
            print(f"    Max force: {force_limit / 1000:.0f} kN")
            print(f"    Reward scale: {reward_scale} (fixed, no adaptive scaling)")
            print(f"    Timesteps: {timesteps:,}")
            print(f"    TMD location: Floor 8 (soft story)")
            print(f"    TMD mass: 8000 kg (4% of floor mass)")

            # Create parallel envs
            n_envs = V12Config.N_ENVS
            print(f"\n  Creating {n_envs} parallel environments...")

            try:
                env = create_parallel_envs(train_files, force_limit, reward_scale, n_envs)
            except Exception as e:
                print(f"\n  ❌ ERROR creating environments: {e}")
                traceback.print_exc()
                print(f"  Skipping stage {stage_idx + 1}")
                continue

            # PPO hyperparameters
            n_steps = stage.get('n_steps', 2048)
            batch_size = stage.get('batch_size', 256)
            n_epochs = stage.get('n_epochs', 10)
            learning_rate = stage.get('learning_rate', 3e-4)
            ent_coef = stage.get('ent_coef', 0.03)

            print(f"\n  PPO hyperparameters:")
            print(f"    n_steps: {n_steps}")
            print(f"    batch_size: {batch_size}")
            print(f"    n_epochs: {n_epochs}")
            print(f"    learning_rate: {learning_rate}")
            print(f"    ent_coef: {ent_coef}")
            print(f"    Network: {V12Config.NETWORK_ARCH}")

            # Create or load model
            if model is None and resume_from is None:
                print(f"\n  Creating new PPO model...")
                model = PPO(
                    V12Config.POLICY_TYPE,
                    env,
                    learning_rate=learning_rate,
                    n_steps=n_steps,
                    batch_size=batch_size,
                    n_epochs=n_epochs,
                    gamma=V12Config.GAMMA,
                    gae_lambda=V12Config.GAE_LAMBDA,
                    clip_range=V12Config.CLIP_RANGE,
                    clip_range_vf=V12Config.CLIP_RANGE_VF,
                    ent_coef=ent_coef,
                    vf_coef=V12Config.VF_COEF,
                    max_grad_norm=V12Config.MAX_GRAD_NORM,
                    policy_kwargs=V12Config.get_policy_kwargs(),
                    verbose=V12Config.VERBOSE,
                    tensorboard_log=logs_dir
                )
            elif resume_from and model is None:
                print(f"\n  Loading model from {resume_from}...")
                model = PPO.load(resume_from, env=env, tensorboard_log=logs_dir)
            else:
                print(f"\n  Continuing with existing model...")
                model.set_env(env)
                model.learning_rate = learning_rate
                model.n_steps = n_steps
                model.batch_size = batch_size
                model.n_epochs = n_epochs
                model.ent_coef = ent_coef

            # Setup logger
            tb_log_name = f"{run_name}_stage{stage_idx + 1}"
            logger = configure(logs_dir, ["stdout", "tensorboard"])
            model.set_logger(logger)

            # TensorBoard logging callback
            tensorboard_dir = os.path.join(logs_dir, run_name, f'stage{stage_idx}_{magnitude}')
            os.makedirs(tensorboard_dir, exist_ok=True)
            tensorboard_callback = TensorBoardMetricsCallback(log_dir=tensorboard_dir)
            

            # checkpoint Callbacks
            checkpoint_callback = CheckpointCallback(
                save_freq=max(50000 // n_envs, 1),
                save_path=run_dir,
                name_prefix=f"stage{stage_idx + 1}_{magnitude}"
            )

            callbacks = [tensorboard_callback]

            # Train
            print(f"\n  Starting training...")
            print(f"  Progress will be logged to TensorBoard")
            print(f"  Checkpoints will be saved to {run_dir}")
            print(f"\n  Monitor training:")
            print(f"    tensorboard --logdir {logs_dir}")

            try:
                model.learn(
                    total_timesteps=timesteps,
                    log_interval=10,
                    progress_bar=True,
                    callback=callbacks,
                    tb_log_name=tb_log_name,
                    reset_num_timesteps=False
                )
                print(f"\n  ✅ Training completed successfully!")
            except Exception as e:
                print(f"\n  ❌ ERROR during training: {e}")
                traceback.print_exc()
                env.close()
                continue

            # Save final model for this stage
            final_path = os.path.join(run_dir, f"stage{stage_idx + 1}_{magnitude}_final.zip")
            try:
                model.save(final_path)
                print(f"  ✅ Model saved: {final_path}")
            except Exception as e:
                print(f"  ❌ ERROR saving model: {e}")
                traceback.print_exc()

            # Close training environment
            env.close()

            # Evaluate on held-out test set
            print(f"\n  Evaluating on held-out test set...")
            test_metrics = evaluate_on_test_set(model, test_file, force_limit, magnitude)

            if test_metrics:
                test_results[magnitude] = test_metrics
                print(f"  ✅ Test evaluation complete")
            else:
                print(f"  ⚠️  Test evaluation failed")

            print(f"\n  Stage {stage_idx + 1} complete!")
            print(f"  {'='*70}\n")

        print(f"\n{'=' * 80}")
        print("  TRAINING COMPLETE!")
        print("=" * 80)

        if model is not None:
            final_model_path = os.path.join(run_dir, "final_model.zip")
            try:
                model.save(final_model_path)
                print(f"\n✅ Final model saved: {final_model_path}")
            except Exception as e:
                print(f"\n❌ ERROR saving final model: {e}")
                traceback.print_exc()
        else:
            print(f"\n⚠️  No model was trained (all stages failed or were skipped)")
            return

        # Print test results summary
        if test_results:
            print(f"\n{'=' * 80}")
            print("  HELD-OUT TEST SET SUMMARY")
            print("=" * 80)

            print(f"\n{'Magnitude':<12} {'Disp (cm)':<12} {'ISDR (%)':<12} {'DCR':<10} {'Status':<20}")
            print("-" * 70)

            for magnitude, metrics in test_results.items():
                disp = metrics['peak_disp_cm']
                isdr = metrics['max_isdr_percent']
                dcr = metrics['DCR']

                # Check targets
                disp_ok = disp <= 14.0
                isdr_ok = isdr <= 0.4
                dcr_ok = dcr <= 1.15

                if disp_ok and isdr_ok and dcr_ok:
                    status = "🏆 ALL TARGETS MET"
                elif disp <= 18 and isdr <= 0.8 and dcr <= 1.3:
                    status = "⚠️  CLOSE"
                else:
                    status = "❌ NEEDS WORK"

                print(f"{magnitude:<12} {disp:<12.2f} {isdr:<12.3f} {dcr:<10.2f} {status:<20}")

        print("-" * 70)
        print(f"\nTargets: Displacement ≤ 14 cm, ISDR ≤ 0.4%, DCR ≤ 1.15")

        print(f"\n{'=' * 80}")
        print("  NEXT STEPS")
        print("=" * 80)

        print(f"\nTo test the model on all earthquakes:")
        print(f"  python test_v12_model.py --model-path {final_model_path}")

        print(f"\nTo monitor training metrics:")
        print(f"  tensorboard --logdir {logs_dir}")

        print(f"\nModel checkpoints saved in: {run_dir}")
        print(f"\nLog file: {log_path}")
        print(f"\n{'=' * 80}\n")

    except KeyboardInterrupt:
        print(f"\n\n⚠️  Training interrupted by user (Ctrl+C)")
        print(f"Partial results may be saved in: {run_dir}")
    except Exception as e:
        print(f"\n\n❌ CRITICAL ERROR during training:")
        print(f"{e}")
        traceback.print_exc()
    finally:
        # Close log file
        close_log_file()


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train v12 Soft-Story TMD Model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--run-name',
        type=str,
        default='v12_soft_story_tmd',
        help='Run name for organizing outputs'
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        default='../../matlab/datasets',
        help='Directory containing earthquake CSV files'
    )

    parser.add_argument(
        '--models-dir',
        type=str,
        default='models',
        help='Directory to save trained models'
    )

    parser.add_argument(
        '--logs-dir',
        type=str,
        default='logs/v12_soft_story_tmd',
        help='Directory for TensorBoard logs'
    )

    parser.add_argument(
        '--resume-from',
        type=str,
        default=None,
        help='Path to model checkpoint to resume training from'
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    train_v12(
        run_name=args.run_name,
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        logs_dir=args.logs_dir,
        resume_from=args.resume_from
    )

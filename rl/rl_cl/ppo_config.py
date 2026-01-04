"""
PPO Training Configuration for Earthquake Control
=================================================

This module defines all hyperparameters and training configurations
for the v8 PPO optimized training script.

All configurations are based on empirical results from v5-v7 training
and PPO-specific optimizations for earthquake control tasks.

Author: Siddharth
Date: January 2026
"""

import torch
from typing import Dict, List, Any


# ============================================================================
# HARDWARE CONFIGURATION
# ============================================================================

def get_device_config() -> Dict[str, Any]:
    """
    Auto-detect and configure compute device (GPU/CPU)

    Returns:
        dict with 'device' (str) and 'gpu_name' (str or None)
    """
    if torch.cuda.is_available():
        return {
            'device': 'cuda',
            'gpu_name': torch.cuda.get_device_name(0),
            'cuda_available': True
        }
    else:
        return {
            'device': 'cpu',
            'gpu_name': None,
            'cuda_available': False
        }


# ============================================================================
# PPO ALGORITHM HYPERPARAMETERS
# ============================================================================

class PPOHyperparameters:
    """
    PPO algorithm hyperparameters

    These are carefully tuned for earthquake control with curriculum learning.
    """

    # Policy architecture
    POLICY_TYPE = "MlpPolicy"
    NETWORK_ARCH = [256, 256]  # Two hidden layers with 256 units each
    ACTIVATION_FN = torch.nn.ReLU

    # Fixed hyperparameters (same across all stages)
    BATCH_SIZE = 256           # Large batches for stability
    N_EPOCHS = 10              # PPO optimization epochs per update
    GAMMA = 0.99               # Discount factor
    GAE_LAMBDA = 0.95          # Generalized Advantage Estimation
    CLIP_RANGE = 0.2           # PPO clipping parameter
    CLIP_RANGE_VF = 0.2        # Value function clipping (prevents divergence)
    VF_COEF = 0.5              # Value function loss coefficient
    MAX_GRAD_NORM = 0.5        # Gradient clipping (prevents exploding gradients)
    VERBOSE = 1                # Enable training logs

    # Parallel environments
    N_ENVS = 4                 # Number of parallel environments

    @staticmethod
    def get_base_config(device: str = 'cpu') -> Dict[str, Any]:
        """
        Get base PPO configuration dict

        Args:
            device: 'cuda' or 'cpu'

        Returns:
            Base PPO configuration
        """
        return {
            'policy': PPOHyperparameters.POLICY_TYPE,
            'batch_size': PPOHyperparameters.BATCH_SIZE,
            'n_epochs': PPOHyperparameters.N_EPOCHS,
            'gamma': PPOHyperparameters.GAMMA,
            'gae_lambda': PPOHyperparameters.GAE_LAMBDA,
            'clip_range': PPOHyperparameters.CLIP_RANGE,
            'clip_range_vf': PPOHyperparameters.CLIP_RANGE_VF,
            'vf_coef': PPOHyperparameters.VF_COEF,
            'max_grad_norm': PPOHyperparameters.MAX_GRAD_NORM,
            'policy_kwargs': {
                'net_arch': PPOHyperparameters.NETWORK_ARCH,
                'activation_fn': PPOHyperparameters.ACTIVATION_FN
            },
            'verbose': PPOHyperparameters.VERBOSE,
            'device': device
        }


# ============================================================================
# CURRICULUM LEARNING STAGES
# ============================================================================

class CurriculumStages:
    """
    Curriculum learning configuration for earthquake control

    Training progresses through 4 stages of increasing difficulty,
    with adaptive hyperparameters per stage.
    """

    # Stage definitions
    STAGES = [
        {
            'name': 'M4.5 @ 50kN',
            'magnitude': 'M4.5',
            'force_limit': 50_000,      # 50 kN max force
            'timesteps': 300_000,       # 2× more than SAC (PPO needs more data)
            'n_steps': 1024,            # Short episodes (~20s)
            'learning_rate': 3e-4,      # Standard LR for easy task
            'ent_coef': 0.02,           # High exploration
            'description': 'Small earthquake - learn basic control'
        },
        {
            'name': 'M5.7 @ 100kN',
            'magnitude': 'M5.7',
            'force_limit': 100_000,     # 100 kN max force
            'timesteps': 300_000,       # 2× more than SAC
            'n_steps': 2048,            # Medium episodes (~40s)
            'learning_rate': 3e-4,      # Standard LR
            'ent_coef': 0.01,           # Medium exploration
            'description': 'Moderate earthquake - refine control strategy'
        },
        {
            'name': 'M7.4 @ 150kN',
            'magnitude': 'M7.4',
            'force_limit': 150_000,     # 150 kN max force
            'timesteps': 400_000,       # 2× more than SAC (critical stage)
            'n_steps': 4096,            # Long episodes (~60s)
            'learning_rate': 1e-4,      # LOWER LR (prevent instability)
            'ent_coef': 0.005,          # Low exploration
            'description': 'High earthquake - careful learning, prevent catastrophic forgetting'
        },
        {
            'name': 'M8.4 @ 150kN',
            'magnitude': 'M8.4',
            'force_limit': 150_000,     # 150 kN max force
            'timesteps': 400_000,       # 2× more than SAC (extreme case)
            'n_steps': 4096,            # Very long episodes (~120s)
            'learning_rate': 5e-5,      # VERY LOW LR (extreme care)
            'ent_coef': 0.001,          # Minimal exploration (exploitation mode)
            'description': 'Extreme earthquake - ultra-careful learning'
        }
    ]

    @staticmethod
    def get_total_timesteps() -> int:
        """Calculate total training timesteps across all stages"""
        return sum(stage['timesteps'] for stage in CurriculumStages.STAGES)

    @staticmethod
    def get_effective_samples(n_envs: int = 4) -> int:
        """Calculate effective number of samples with parallel envs"""
        return CurriculumStages.get_total_timesteps() * n_envs

    @staticmethod
    def print_curriculum_summary():
        """Print curriculum learning plan"""
        print("\n🎯 Curriculum Learning Plan:")
        print("="*70)
        for i, stage in enumerate(CurriculumStages.STAGES, 1):
            print(f"\nStage {i}: {stage['name']}")
            print(f"  Description: {stage['description']}")
            print(f"  Timesteps: {stage['timesteps']:,}")
            print(f"  n_steps: {stage['n_steps']}")
            print(f"  Learning rate: {stage['learning_rate']:.0e}")
            print(f"  Entropy coef: {stage['ent_coef']}")
            print(f"  Force limit: {stage['force_limit']:,} N")

        print(f"\n{'='*70}")
        print(f"Total timesteps: {CurriculumStages.get_total_timesteps():,}")
        print(f"Effective samples (4 envs): {CurriculumStages.get_effective_samples():,}")
        print(f"{'='*70}\n")


# ============================================================================
# ADAPTIVE REWARD SCALING
# ============================================================================

class AdaptiveRewardConfig:
    """
    Magnitude-adaptive reward scaling configuration

    Based on v7 results: different magnitudes need different reward scales
    to prevent gradient instability.
    """

    REWARD_SCALES = {
        'M4.5': 3.0,   # Gentle (avoid over-penalizing small displacements)
        'M5.7': 7.0,   # Strong (showed best improvement in v6)
        'M7.4': 4.0,   # Balanced (10× caused instability, 5× marginal)
        'M8.4': 3.0,   # Conservative (extreme earthquakes need gentler signal)
    }

    @staticmethod
    def get_reward_scale(magnitude: str) -> float:
        """Get reward scale for a given magnitude"""
        return AdaptiveRewardConfig.REWARD_SCALES.get(magnitude, 1.0)

    @staticmethod
    def print_reward_strategy():
        """Print adaptive reward scaling strategy"""
        print("\n🎯 Adaptive Reward Scaling:")
        print("="*70)
        for mag, scale in AdaptiveRewardConfig.REWARD_SCALES.items():
            print(f"  {mag}: {scale}× multiplier")
        print(f"{'='*70}\n")


# ============================================================================
# TRAINING DATA CONFIGURATION
# ============================================================================

class TrainingDataConfig:
    """
    Training and test data configuration

    Proper train/test split to prevent memorization.
    """

    # Training data (synthetic variants)
    TRAIN_DIR = "../../matlab/datasets/training_set_v2"
    N_VARIANTS_PER_MAGNITUDE = 10  # 10 synthetic variants per magnitude

    # Test data (held-out PEER earthquakes)
    TEST_FILES = {
        'M4.5': "../../matlab/datasets/PEER_small_M4.5_PGA0.25g.csv",
        'M5.7': "../../matlab/datasets/PEER_moderate_M5.7_PGA0.35g.csv",
        'M7.4': "../../matlab/datasets/PEER_high_M7.4_PGA0.75g.csv",
        'M8.4': "../../matlab/datasets/PEER_insane_M8.4_PGA0.9g.csv"
    }

    # Uncontrolled baselines (for comparison)
    UNCONTROLLED_BASELINES = {
        'M4.5': 21.02,   # cm
        'M5.7': 46.02,   # cm
        'M7.4': 235.55,  # cm
        'M8.4': 357.06   # cm
    }

    @staticmethod
    def print_data_strategy():
        """Print dataset strategy"""
        print("\n📊 Dataset Strategy:")
        print("="*70)
        print(f"TRAINING: {TrainingDataConfig.TRAIN_DIR}")
        print(f"  → {TrainingDataConfig.N_VARIANTS_PER_MAGNITUDE} variants per magnitude")
        print("\nTESTING: Held-out PEER earthquakes")
        for mag, path in TrainingDataConfig.TEST_FILES.items():
            print(f"  → {mag}: {path.split('/')[-1]}")
        print(f"{'='*70}\n")


# ============================================================================
# MODEL SAVING CONFIGURATION
# ============================================================================

class ModelSavingConfig:
    """Model saving and checkpointing configuration"""

    OUTPUT_DIR = "models/rl_v8_ppo_optimized"

    # Checkpoint naming
    STAGE_CHECKPOINT_TEMPLATE = "stage{stage_num}_{force_kn}kN.zip"
    FINAL_MODEL_NAME = "final_v8_ppo_optimized.zip"

    @staticmethod
    def get_stage_checkpoint_path(stage_num: int, force_limit: int) -> str:
        """Get checkpoint path for a given stage"""
        force_kn = force_limit // 1000
        filename = ModelSavingConfig.STAGE_CHECKPOINT_TEMPLATE.format(
            stage_num=stage_num,
            force_kn=force_kn
        )
        return f"{ModelSavingConfig.OUTPUT_DIR}/{filename}"

    @staticmethod
    def get_final_model_path() -> str:
        """Get final model path"""
        return f"{ModelSavingConfig.OUTPUT_DIR}/{ModelSavingConfig.FINAL_MODEL_NAME}"


# ============================================================================
# COMPARISON BASELINES
# ============================================================================

class ComparisonBaselines:
    """Previous model results for comparison"""

    V5_RESULTS = {
        'M4.5': 20.73,
        'M5.7': 44.84,
        'M7.4': 229.20,
        'M8.4': 366.04
    }

    V6_RESULTS = {
        'M4.5': 20.97,
        'M5.7': 41.31,
        'M7.4': 260.37,
        'M8.4': None
    }

    V7_SAC_RESULTS = {
        'M4.5': 20.72,
        'M5.7': 46.45,
        'M7.4': 219.30,  # BEST SAC result
        'M8.4': 363.36
    }

    @staticmethod
    def get_comparison_summary(magnitude: str, v8_result: float) -> str:
        """Get comparison summary string for a magnitude"""
        v5 = ComparisonBaselines.V5_RESULTS.get(magnitude, 0)
        v6 = ComparisonBaselines.V6_RESULTS.get(magnitude)
        v7_sac = ComparisonBaselines.V7_SAC_RESULTS.get(magnitude, 0)

        summary = f"\n{magnitude} Comparison:\n"
        summary += f"  v5 (5× uniform):       {v5:.2f} cm\n"
        if v6:
            summary += f"  v6 (10× uniform):      {v6:.2f} cm\n"
        summary += f"  v7-SAC (adaptive):     {v7_sac:.2f} cm\n"
        summary += f"  v8-PPO-OPT (adaptive): {v8_result:.2f} cm"

        if v8_result < v7_sac:
            delta = v7_sac - v8_result
            summary += f"  🏆 BEATS v7-SAC by {delta:.2f} cm!"

        return summary


# ============================================================================
# MAIN CONFIG CLASS
# ============================================================================

class V8PPOConfig:
    """
    Complete v8 PPO training configuration

    This class aggregates all configuration components for easy access.
    """

    def __init__(self):
        self.device_config = get_device_config()
        self.hyperparameters = PPOHyperparameters
        self.curriculum = CurriculumStages
        self.rewards = AdaptiveRewardConfig
        self.data = TrainingDataConfig
        self.saving = ModelSavingConfig
        self.baselines = ComparisonBaselines

    def print_full_config(self):
        """Print complete configuration summary"""
        print("\n" + "="*70)
        print("  V8 PPO OPTIMIZED TRAINING CONFIGURATION")
        print("="*70)

        # Device info
        if self.device_config['cuda_available']:
            print(f"\n🚀 GPU: {self.device_config['gpu_name']}")
            print(f"   Device: {self.device_config['device'].upper()}")
        else:
            print(f"\n⚠️  Device: {self.device_config['device'].upper()} (No GPU detected)")

        # PPO hyperparameters
        print("\n🤖 PPO Hyperparameters:")
        print(f"   Network: {self.hyperparameters.NETWORK_ARCH} with {self.hyperparameters.ACTIVATION_FN.__name__}")
        print(f"   Batch size: {self.hyperparameters.BATCH_SIZE}")
        print(f"   Epochs: {self.hyperparameters.N_EPOCHS}")
        print(f"   Gamma: {self.hyperparameters.GAMMA}")
        print(f"   GAE lambda: {self.hyperparameters.GAE_LAMBDA}")
        print(f"   Clip range: {self.hyperparameters.CLIP_RANGE}")
        print(f"   Value clip: {self.hyperparameters.CLIP_RANGE_VF}")
        print(f"   Grad clip: {self.hyperparameters.MAX_GRAD_NORM}")
        print(f"   Parallel envs: {self.hyperparameters.N_ENVS}")

        # Curriculum
        self.curriculum.print_curriculum_summary()

        # Rewards
        self.rewards.print_reward_strategy()

        # Data
        self.data.print_data_strategy()

        print("="*70 + "\n")


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Create and display configuration
    config = V8PPOConfig()
    config.print_full_config()

    # Example: Access specific configs
    print("\nExample config access:")
    print(f"Device: {config.device_config['device']}")
    print(f"Total timesteps: {config.curriculum.get_total_timesteps():,}")
    print(f"M7.4 reward scale: {config.rewards.get_reward_scale('M7.4')}×")
    print(f"Final model path: {config.saving.get_final_model_path()}")

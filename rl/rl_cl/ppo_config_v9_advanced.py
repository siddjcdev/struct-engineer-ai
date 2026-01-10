"""
PPO v9 Advanced Configuration - Enhanced Hyperparameters
=========================================================

This module implements advanced PPO optimizations based on empirical
analysis and best practices for earthquake control.

Key Improvements over v8:
1. Smoother learning rate transitions with cosine annealing
2. Refined entropy coefficient scheduling
3. Increased n_steps for better advantage estimates
4. Balanced batch_size and n_epochs for stability
5. Deeper network architecture
6. Optimized value function clipping

Author: Siddharth
Date: January 2026
"""

import torch
import numpy as np
from typing import Dict, List, Any, Callable


# ============================================================================
# LEARNING RATE SCHEDULES
# ============================================================================

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule

    Args:
        initial_value: Initial learning rate

    Returns:
        Schedule function that takes progress (0-1) and returns LR
    """
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


def cosine_schedule(initial_value: float, final_value: float = 0.0) -> Callable[[float], float]:
    """
    Cosine annealing learning rate schedule

    Smoother decay than linear, often leads to better convergence.

    Args:
        initial_value: Initial learning rate
        final_value: Final learning rate (default: 0.0)

    Returns:
        Schedule function that takes progress (0-1) and returns LR
    """
    def func(progress_remaining: float) -> float:
        # progress_remaining goes from 1.0 to 0.0
        # We want to go from initial to final
        progress = 1.0 - progress_remaining
        cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
        return final_value + (initial_value - final_value) * cosine_decay
    return func


# ============================================================================
# ADVANCED PPO HYPERPARAMETERS
# ============================================================================

class V9PPOHyperparameters:
    """
    Advanced PPO hyperparameters for v9

    Key improvements:
    - Larger n_steps for reduced variance
    - Balanced batch_size and n_epochs
    - Deeper network architecture
    - Optimized clipping ranges
    """

    # Policy architecture - ENHANCED (deeper network)
    POLICY_TYPE = "MlpPolicy"
    NETWORK_ARCH = [256, 256, 256, 256]  # FOUR layers (was three earlier)
    ACTIVATION_FN = torch.nn.Tanh  # Tanh works better for earthquake control (bounded actions)

    # Learning optimization
    GAMMA = 0.99               # Discount factor
    GAE_LAMBDA = 0.95          # Generalized Advantage Estimation

    # PPO clipping
    CLIP_RANGE = 0.2           # Policy clipping
    CLIP_RANGE_VF = 0.15       # REFINED: Slightly tighter for value (was 0.2)

    # Regularization
    VF_COEF = 0.5              # Value function loss coefficient
    MAX_GRAD_NORM = 0.5        # Gradient clipping
    VERBOSE = 1                # Enable training logs

    # Parallel environments
    N_ENVS = 4                 # Number of parallel environments

    @staticmethod
    def get_policy_kwargs(use_advanced_arch: bool = True) -> Dict[str, Any]:
        """
        Get policy network configuration

        Args:
            use_advanced_arch: Use deeper network with optimized activations

        Returns:
            Policy kwargs dict
        """
        if use_advanced_arch:
            return {
                'net_arch': V9PPOHyperparameters.NETWORK_ARCH,
                'activation_fn': V9PPOHyperparameters.ACTIVATION_FN
            }
        else:
            # Fallback to v8 architecture
            return {
                'net_arch': [256, 256],
                'activation_fn': torch.nn.ReLU
            }


# ============================================================================
# ADVANCED CURRICULUM STAGES
# ============================================================================

class V9CurriculumStages:
    """
    Advanced curriculum with refined hyperparameters per stage

    Key improvements:
    - Smoother LR transitions
    - Increased n_steps for better advantage estimates
    - Balanced batch_size and n_epochs
    - Refined entropy scheduling
    """

    STAGES = [
        {
            'name': 'M4.5 @ 150kN - Extended',
            'magnitude': 'M4.5',
            'force_limit': 150_000,    # 150 kN - sufficient control authority
            'timesteps': 1_000_000,    # 1M steps for aggressive targets (14cm, 0.4% ISDR)
            'reward_scale': 1.0,       # CRITICAL: Fixed reward scale (no adaptive scaling!)

            # PPO parameters - BALANCED
            'n_steps': 2048,           # INCREASED from 1024 (better advantage estimates)
            'batch_size': 256,         # Balanced with n_steps
            'n_epochs': 10,            # Standard

            # Learning schedule - ENABLE FOR LONG TRAINING
            'learning_rate': 3e-4,     # Start at 3e-4
            'use_lr_schedule': True,   # Enable cosine decay for 1M steps
            'final_lr': 1e-4,          # Decay to 1e-4 for fine-tuning

            # Exploration - MODERATE
            'ent_coef': 0.03,          # Balanced (between 0.02-0.05)
            'ent_schedule': False,     # Fixed entropy

            'description': 'Extended training (1M steps) with fixed reward_scale=1.0 for aggressive targets'
        },
        # {
        #     'name': 'M5.7 @ 100kN',
        #     'magnitude': 'M5.7',
        #     'force_limit': 100_000,
        #     'timesteps': 300_000,

        #     # PPO parameters - BALANCED
        #     'n_steps': 4096,           # INCREASED from 2048 (longer rollouts)
        #     'batch_size': 512,         # INCREASED to balance larger n_steps
        #     'n_epochs': 12,            # INCREASED slightly (more robust updates)

        #     # Learning schedule - COSINE ANNEALING
        #     'learning_rate': 3e-4,     # Start same as stage 1
        #     'use_lr_schedule': True,   # Enable cosine annealing
        #     'final_lr': 2e-4,          # Smooth decay to 2e-4

        #     # Exploration - MEDIUM-HIGH
        #     'batch_size': 256,         # Balanced with n_steps ,from 256 to 512 #Did not work well
        #     'ent_coef': 0.015,         # Moderate exploration, FROM 0.015 to 0.075 
        #     'ent_schedule': False,

        #     'description': 'Moderate earthquake - longer rollouts with cosine LR decay'
        # },
        # {
        #     'name': 'M7.4 @ 150kN',
        #     'magnitude': 'M7.4',
        #     'force_limit': 150_000,
        #     'timesteps': 400_000,

        #     # PPO parameters - LARGER BUFFERS
        #     'n_steps': 8192,           # SIGNIFICANTLY INCREASED (reduce variance)
        #     'batch_size': 512,         # Keep large for stability
        #     'n_epochs': 15,            # INCREASED (more thorough updates)

        #     # Learning schedule - CAREFUL DECAY
        #     'learning_rate': 2e-4,     # SMOOTHER transition from stage 2 (was 1e-4)
        #     'use_lr_schedule': True,
        #     'final_lr': 1e-4,          # Decay to 1e-4

        #     # Exploration - LOW
        #     'ent_coef': 0.008,         # SLIGHTLY HIGHER than v8 (0.005) for initial exploration, FROM 0.008 to 0.08 DID NOT WORK
        #     'ent_schedule': True,      # Anneal during training
        #     'final_ent': 0.003,

        #     'description': 'High earthquake - large rollouts, careful learning, prevent catastrophic forgetting'
        # },
        # {
        #     'name': 'M8.4 @ 150kN',
        #     'magnitude': 'M8.4',
        #     'force_limit': 150_000,
        #     'timesteps': 400_000,

        #     # PPO parameters - MAXIMUM STABILITY
        #     'n_steps': 8192,           # Keep large (longest episodes)
        #     'batch_size': 512,         # Large batches for stability
        #     'n_epochs': 20,            # MAXIMUM epochs (most thorough updates)

        #     # Learning schedule - ULTRA-CAREFUL
        #     'learning_rate': 1e-4,     # Smooth continuation from stage 3
        #     'use_lr_schedule': True,
        #     'final_lr': 5e-5,          # Very low final LR

        #     # Exploration - MINIMAL
        #     'ent_coef': 0.005,         # SLIGHTLY HIGHER than v8 (0.001) FROM 0.005 to 0.05 DID NOT WORK
        #     'ent_schedule': True,
        #     'final_ent': 0.001,        # Decay to exploitation

        #     'description': 'Extreme earthquake - maximum stability, ultra-careful learning'
        # }
    ]

    @staticmethod
    def get_total_timesteps() -> int:
        """Calculate total training timesteps"""
        return sum(stage['timesteps'] for stage in V9CurriculumStages.STAGES)

    @staticmethod
    def get_effective_samples(n_envs: int = 4) -> int:
        """Calculate effective samples with parallel envs"""
        return V9CurriculumStages.get_total_timesteps() * n_envs

    @staticmethod
    def print_curriculum_summary():
        """Print detailed curriculum plan"""
        print("\n🎯 V9 Advanced Curriculum Plan:")
        print("="*70)
        for i, stage in enumerate(V9CurriculumStages.STAGES, 1):
            print(f"\nStage {i}: {stage['name']}")
            print(f"  Description: {stage['description']}")
            print(f"  Timesteps: {stage['timesteps']:,}")
            print(f"  n_steps: {stage['n_steps']} (rollout buffer size)")
            print(f"  batch_size: {stage['batch_size']}")
            print(f"  n_epochs: {stage['n_epochs']}")
            print(f"  Learning rate: {stage['learning_rate']:.0e}", end='')
            if stage['use_lr_schedule']:
                print(f" → {stage['final_lr']:.0e} (cosine decay)")
            else:
                print(" (fixed)")
            print(f"  Entropy coef: {stage['ent_coef']}", end='')
            if stage.get('ent_schedule'):
                print(f" → {stage['final_ent']} (annealing)")
            else:
                print(" (fixed)")

        print(f"\n{'='*70}")
        print(f"Total timesteps: {V9CurriculumStages.get_total_timesteps():,}")
        print(f"Effective samples (4 envs): {V9CurriculumStages.get_effective_samples():,}")
        print(f"{'='*70}\n")

    @staticmethod
    def get_learning_rate_schedule(stage: Dict) -> Callable:
        """
        Get learning rate schedule for a stage

        Args:
            stage: Stage configuration dict

        Returns:
            Learning rate schedule function or float
        """
        if stage.get('use_lr_schedule', False):
            return cosine_schedule(
                stage['learning_rate'],
                stage.get('final_lr', stage['learning_rate'] * 0.1)
            )
        else:
            return stage['learning_rate']


# ============================================================================
# V9 CONFIGURATION SUMMARY
# ============================================================================

class V9AdvancedConfig:
    """
    Complete v9 advanced configuration

    Key improvements over v8:
    1. Deeper network (3 layers vs 2)
    2. Larger n_steps (2048-8192 vs 1024-4096)
    3. Balanced batch_size and n_epochs
    4. Smoother learning rate transitions
    5. Refined entropy scheduling
    6. Tighter value function clipping
    """

    VERSION = "v9-advanced"
    DESCRIPTION = "Advanced PPO with refined hyperparameters and scheduling"

    # Configuration components
    hyperparameters = V9PPOHyperparameters
    curriculum = V9CurriculumStages

    @classmethod
    def print_improvements_summary(cls):
        """Print summary of improvements over v8"""
        print("\n" + "="*70)
        print("  V9 ADVANCED IMPROVEMENTS OVER V8")
        print("="*70)
        print("\n1. Network Architecture:")
        print("   • Deeper: 3 layers [256, 256, 256] (was 2 layers)")
        print("   • Tanh output activation for bounded actions")
        print("\n2. Rollout Buffer (n_steps):")
        print("   • M4.5: 2048 (was 1024) - 2× larger")
        print("   • M5.7: 4096 (was 2048) - 2× larger")
        print("   • M7.4: 8192 (was 4096) - 2× larger")
        print("   • M8.4: 8192 (was 4096) - 2× larger")
        print("   → Reduces variance in advantage estimates")
        print("\n3. Batch Size & Epochs:")
        print("   • M4.5: batch=256, epochs=10 (unchanged)")
        print("   • M5.7: batch=512, epochs=12 (was 256/10)")
        print("   • M7.4: batch=512, epochs=15 (was 256/10)")
        print("   • M8.4: batch=512, epochs=20 (was 256/10)")
        print("   → Better balance for larger n_steps")
        print("\n4. Learning Rate Schedule:")
        print("   • Cosine annealing within stages (smoother decay)")
        print("   • Smoother transitions between stages:")
        print("     - Stage 1: 3e-4 (fixed)")
        print("     - Stage 2: 3e-4 → 2e-4 (cosine)")
        print("     - Stage 3: 2e-4 → 1e-4 (cosine)")
        print("     - Stage 4: 1e-4 → 5e-5 (cosine)")
        print("\n5. Entropy Scheduling:")
        print("   • Slightly higher initial values for better exploration")
        print("   • Annealing in hard stages (M7.4, M8.4)")
        print("\n6. Value Function Clipping:")
        print("   • Tighter: 0.15 (was 0.2) - more stable value learning")
        print("\n" + "="*70 + "\n")


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print("\n🚀 V9 Advanced PPO Configuration\n")

    # Print improvements
    V9AdvancedConfig.print_improvements_summary()

    # Print curriculum
    V9AdvancedConfig.curriculum.print_curriculum_summary()

    # Example: Get stage configuration
    stage = V9AdvancedConfig.curriculum.STAGES[2]  # M7.4
    print(f"\nExample - Stage 3 (M7.4) Configuration:")
    print(f"  n_steps: {stage['n_steps']}")
    print(f"  batch_size: {stage['batch_size']}")
    print(f"  n_epochs: {stage['n_epochs']}")
    print(f"  learning_rate: {stage['learning_rate']:.0e} → {stage['final_lr']:.0e}")
    print(f"  ent_coef: {stage['ent_coef']} → {stage['final_ent']}")

    # Get LR schedule
    lr_schedule = V9AdvancedConfig.curriculum.get_learning_rate_schedule(stage)
    print(f"\n  Learning rate at different progress points:")
    for progress in [0.0, 0.25, 0.5, 0.75, 1.0]:
        lr = lr_schedule(progress)
        print(f"    Progress {int((1-progress)*100)}%: {lr:.2e}")

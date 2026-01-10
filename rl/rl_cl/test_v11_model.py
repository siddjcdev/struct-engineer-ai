"""
Test Script for v11 Advanced PPO Model
========================================

This script evaluates a trained v11 model on all test earthquakes and provides:
- Peak displacement, ISDR, and DCR metrics
- Comparison with uncontrolled baseline
- Comparison with v8 baseline (if available)
- Performance summary across all magnitudes

Usage:
    python test_v11_model.py --model-path models/rl_v11_advanced/final_1M_fixed_scale/stage1_150kN.zip
    python test_v11_model.py --model-path models/rl_v11_advanced/final_1M_fixed_scale/stage1_150kN.zip --test-dir ../../matlab/datasets

Author: Claude Code
Date: January 2026
"""

import sys
import os
import argparse
import numpy as np
from stable_baselines3 import PPO

# Add restapi path for environment imports
sys.path.insert(0, os.path.abspath('../../restapi/rl_cl'))
from tmd_environment_adaptive_reward import make_improved_tmd_env


# ============================================================================
# BASELINE PERFORMANCE DATA
# ============================================================================

# Uncontrolled baseline (no TMD control, just passive)
UNCONTROLLED_BASELINES = {
    'M4.5': 21.02,   # cm
    'M5.7': 46.02,   # cm
    'M7.4': 235.55,  # cm
    'M8.4': 357.06,  # cm
}

# v8 baseline (if available - update these values if you have them)
V8_BASELINES = {
    'M4.5': 20.72,   # cm
    'M5.7': 46.45,   # cm
    'M7.4': 219.30,  # cm
    'M8.4': 363.36,  # cm
}


# ============================================================================
# TEST FUNCTIONS
# ============================================================================

def test_model_on_earthquake(model, earthquake_file, force_limit=150000, reward_scale=1.0):
    """
    Test trained model on a single earthquake file

    Args:
        model: Trained PPO model
        earthquake_file: Path to earthquake CSV file
        force_limit: Maximum control force in Newtons
        reward_scale: Fixed reward scale (1.0 = no scaling)

    Returns:
        dict: Episode metrics including displacement, ISDR, DCR, forces
    """
    # Create test environment with same settings as training
    env = make_improved_tmd_env(
        earthquake_file,
        max_force=force_limit,
        reward_scale=reward_scale
    )

    # Run episode
    obs, _ = env.reset()
    done = False
    truncated = False

    peak_disp = 0.0
    forces = []

    while not (done or truncated):
        # Get deterministic action from trained policy
        action, _ = model.predict(obs, deterministic=True)

        # Step environment
        obs, reward, done, truncated, info = env.step(action)

        # Track peak displacement
        peak_disp = max(peak_disp, abs(info['roof_displacement']))
        forces.append(abs(info['control_force']))

    # Get final episode metrics
    metrics = env.get_episode_metrics()

    # Add peak displacement in cm
    metrics['peak_disp_cm'] = peak_disp * 100

    # Add force statistics
    metrics['peak_force_kN'] = np.max(forces) / 1000
    metrics['mean_force_kN'] = np.mean(forces) / 1000
    metrics['rms_force_kN'] = np.sqrt(np.mean(np.array(forces)**2)) / 1000

    env.close()

    return metrics


def print_detailed_results(magnitude, metrics, uncontrolled, v8_baseline=None):
    """
    Print detailed test results for a single earthquake

    Args:
        magnitude: Earthquake magnitude (e.g., 'M4.5')
        metrics: Test metrics dict
        uncontrolled: Uncontrolled baseline displacement (cm)
        v8_baseline: v8 baseline displacement (cm), optional
    """
    peak_cm = metrics['peak_disp_cm']
    improvement = 100 * (uncontrolled - peak_cm) / uncontrolled

    print(f"\n{'='*70}")
    print(f"  {magnitude} TEST RESULTS")
    print(f"{'='*70}")

    # Displacement comparison
    print(f"\n  Peak Roof Displacement:")
    print(f"    v11-Advanced: {peak_cm:.2f} cm")

    if v8_baseline is not None:
        delta_v8 = peak_cm - v8_baseline
        v8_status = "🏆 IMPROVED" if delta_v8 < 0 else "❌ WORSE"
        print(f"    v8-Baseline:  {v8_baseline:.2f} cm")
        print(f"    Δ from v8:    {delta_v8:+.2f} cm {v8_status}")

    print(f"    Uncontrolled: {uncontrolled:.2f} cm")

    status = "✓" if improvement > 0 else "✗"
    print(f"    Improvement:  {improvement:+.1f}% {status}")

    # Structural safety metrics
    print(f"\n  Structural Safety:")
    print(f"    Max ISDR:     {metrics['max_isdr_percent']:.2f}%")
    isdr_status = "✅" if metrics['max_isdr_percent'] < 0.5 else "⚠️" if metrics['max_isdr_percent'] < 1.5 else "❌"
    print(f"                  {isdr_status} (target: <0.5%, limit: <1.5%)")

    print(f"    DCR:          {metrics['DCR']:.2f}")
    dcr_status = "✅" if metrics['DCR'] < 1.1 else "⚠️" if metrics['DCR'] < 1.75 else "❌"
    print(f"                  {dcr_status} (target: ~1.0, limit: <1.75)")

    # Control effort
    print(f"\n  Control Effort:")
    print(f"    Peak force:   {metrics['peak_force_kN']:.1f} kN")
    print(f"    Mean force:   {metrics['mean_force_kN']:.1f} kN")
    print(f"    RMS force:    {metrics['rms_force_kN']:.1f} kN")

    # Additional metrics
    print(f"\n  Additional Metrics:")
    print(f"    RMS displacement: {metrics['rms_roof_displacement']*100:.2f} cm")
    print(f"    Max drift:        {metrics['max_drift']*100:.2f} cm")


def print_summary_table(results):
    """
    Print summary table comparing all earthquake magnitudes

    Args:
        results: List of (magnitude, metrics, uncontrolled, v8) tuples
    """
    print(f"\n\n{'='*70}")
    print(f"  SUMMARY: v11 Advanced PPO Performance")
    print(f"{'='*70}\n")

    print(f"{'Magnitude':<12} {'v11 (cm)':<12} {'v8 (cm)':<12} {'Uncont (cm)':<14} {'Improve':<10} {'ISDR%':<10} {'DCR':<8}")
    print(f"{'-'*70}")

    total_improvement = 0
    count = 0

    for magnitude, metrics, uncontrolled, v8 in results:
        peak_cm = metrics['peak_disp_cm']
        improvement = 100 * (uncontrolled - peak_cm) / uncontrolled

        v8_str = f"{v8:.2f}" if v8 is not None else "N/A"
        improve_str = f"{improvement:+.1f}%"

        print(f"{magnitude:<12} {peak_cm:<12.2f} {v8_str:<12} {uncontrolled:<14.2f} {improve_str:<10} {metrics['max_isdr_percent']:<10.2f} {metrics['DCR']:<8.2f}")

        if improvement > 0:
            total_improvement += improvement
            count += 1

    print(f"{'-'*70}")

    if count > 0:
        avg_improvement = total_improvement / count
        print(f"\nAverage improvement over uncontrolled: {avg_improvement:.1f}%")
    else:
        print(f"\n⚠️  No improvements over uncontrolled baseline")


def evaluate_targets(results):
    """
    Evaluate whether aggressive targets were met

    Args:
        results: List of (magnitude, metrics, uncontrolled, v8) tuples
    """
    print(f"\n\n{'='*70}")
    print(f"  TARGET ACHIEVEMENT ANALYSIS")
    print(f"{'='*70}\n")

    # Focus on M4.5 for target evaluation
    m45_result = next((r for r in results if r[0] == 'M4.5'), None)

    if m45_result is None:
        print("⚠️  M4.5 results not available for target analysis")
        return

    _, metrics, _, _ = m45_result

    print("M4.5 Targets (\"Almost No Structural Damage\"):")
    print(f"  Target: 10-18 cm displacement, 0.3-0.5% ISDR, DCR ~1.0\n")

    # Displacement target
    peak_cm = metrics['peak_disp_cm']
    disp_status = "✅ MET" if 10 <= peak_cm <= 18 else "⚠️ CLOSE" if 18 < peak_cm <= 22 else "❌ NOT MET"
    print(f"  Displacement: {peak_cm:.2f} cm {disp_status}")

    # ISDR target
    isdr = metrics['max_isdr_percent']
    isdr_status = "✅ MET" if isdr <= 0.5 else "⚠️ CLOSE" if isdr <= 0.8 else "❌ NOT MET"
    print(f"  ISDR:         {isdr:.2f}% {isdr_status}")

    # DCR target
    dcr = metrics['DCR']
    dcr_status = "✅ MET" if dcr <= 1.1 else "⚠️ CLOSE" if dcr <= 1.3 else "❌ NOT MET"
    print(f"  DCR:          {dcr:.2f} {dcr_status}")

    # Overall assessment
    all_met = (10 <= peak_cm <= 18) and (isdr <= 0.5) and (dcr <= 1.1)
    close = (peak_cm <= 22) and (isdr <= 0.8) and (dcr <= 1.3)

    print(f"\n  Overall Assessment:")
    if all_met:
        print("  ✅ ALL TARGETS MET - Excellent performance!")
        print("     'Almost no structural damage' achieved")
    elif close:
        print("  ⚠️  CLOSE TO TARGETS - Very good performance")
        print("     May indicate physical limits of current TMD configuration")
        print("     Consider: Increase TMD mass (2%→3%) or force (150→200 kN)")
    else:
        print("  ❌ TARGETS NOT MET - More training may be needed")
        print("     Check TensorBoard for convergence issues")
        print("     Ensure training completed full 1M steps")


# ============================================================================
# MAIN TEST FUNCTION
# ============================================================================

def test_v11_model(model_path, test_dir='../../matlab/datasets', force_limit=150000):
    """
    Main test function - evaluates model on all test earthquakes

    Args:
        model_path: Path to trained model .zip file
        test_dir: Directory containing test earthquake files
        force_limit: Maximum control force used during training
    """
    print(f"\n{'='*70}")
    print(f"  TESTING v11 ADVANCED PPO MODEL")
    print(f"{'='*70}\n")

    # Load trained model
    print(f"Loading model: {model_path}")
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found!")
        print(f"   Expected: {model_path}")
        return

    try:
        model = PPO.load(model_path)
        print(f"✅ Model loaded successfully\n")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # Define test earthquakes
    test_files = {
        'M4.5': os.path.join(test_dir, 'PEER_small_M4.5_PGA0.25g.csv'),
        'M5.7': os.path.join(test_dir, 'PEER_moderate_M5.7_PGA0.35g.csv'),
        'M7.4': os.path.join(test_dir, 'PEER_high_M7.4_PGA0.75g.csv'),
        'M8.4': os.path.join(test_dir, 'PEER_insane_M8.4_PGA0.9g.csv'),
    }

    print(f"Test Configuration:")
    print(f"  Force limit: {force_limit/1000:.0f} kN")
    print(f"  Reward scale: 1.0 (fixed, no adaptive scaling)")
    print(f"  Test directory: {test_dir}\n")

    # Test on each earthquake
    results = []

    for magnitude, test_file in test_files.items():
        if not os.path.exists(test_file):
            print(f"⚠️  Skipping {magnitude}: File not found")
            print(f"   Expected: {test_file}\n")
            continue

        print(f"Testing on {magnitude}...")
        print(f"  File: {os.path.basename(test_file)}")

        try:
            # Run test
            metrics = test_model_on_earthquake(
                model,
                test_file,
                force_limit=force_limit,
                reward_scale=1.0  # Use same fixed scale as training
            )

            # Get baselines
            uncontrolled = UNCONTROLLED_BASELINES.get(magnitude, None)
            v8_baseline = V8_BASELINES.get(magnitude, None)

            # Print detailed results
            print_detailed_results(magnitude, metrics, uncontrolled, v8_baseline)

            # Store for summary
            results.append((magnitude, metrics, uncontrolled, v8_baseline))

        except Exception as e:
            print(f"❌ Error testing {magnitude}: {e}\n")
            continue

    # Print summary
    if results:
        print_summary_table(results)
        evaluate_targets(results)
    else:
        print("\n❌ No test results available")

    print(f"\n{'='*70}")
    print(f"  TESTING COMPLETE")
    print(f"{'='*70}\n")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Test v11 Advanced PPO Model on Earthquake Data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to trained model .zip file (e.g., models/rl_v11_advanced/final_1M_fixed_scale/stage1_150kN.zip)'
    )

    parser.add_argument(
        '--test-dir',
        type=str,
        default='../../matlab/datasets',
        help='Directory containing test earthquake CSV files'
    )

    parser.add_argument(
        '--force-limit',
        type=int,
        default=150000,
        help='Maximum control force in Newtons (should match training)'
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    test_v11_model(
        model_path=args.model_path,
        test_dir=args.test_dir,
        force_limit=args.force_limit
    )

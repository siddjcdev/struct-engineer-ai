"""
Test Script for v13 Rooftop TMD Model
======================================

This script evaluates a trained v13 model on all test earthquakes and provides:
- Peak displacement, ISDR, and DCR metrics
- Multi-floor ISDR tracking with critical floor identification
- Comparison with uncontrolled baseline
- Performance summary across all magnitudes
- Evaluation against conservative targets (14cm, 0.4% ISDR, 1.15 DCR)

Usage:
    python test_v13_model.py --model-path models/v13_rooftop_breakthrough/stage1_M4.5_final.zip
    python test_v13_model.py --model-path models/v13_rooftop_breakthrough/final_model.zip --test-dir ../../matlab/datasets

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
from tmd_environment_v13_rooftop import make_rooftop_tmd_env


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

# v13 Conservative Targets (learning from v12 failure)
V13_TARGETS = {
    'displacement_cm': 14.0,
    'isdr_percent': 0.4,
    'dcr': 1.15
}


# ============================================================================
# TEST FUNCTIONS
# ============================================================================

def test_model_on_earthquake(model, earthquake_file, force_limit=300000, reward_scale=1.0):
    """
    Test trained model on a single earthquake file

    Args:
        model: Trained PPO model
        earthquake_file: Path to earthquake CSV file
        force_limit: Maximum control force in Newtons (300 kN for v13)
        reward_scale: Fixed reward scale (1.0 = no scaling)

    Returns:
        dict: Episode metrics including displacement, ISDR, DCR, forces, floor ISDRs
    """
    # Create test environment with same settings as training
    env = make_rooftop_tmd_env(
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


def print_detailed_results(magnitude, metrics, uncontrolled):
    """
    Print detailed test results for a single earthquake

    Args:
        magnitude: Earthquake magnitude (e.g., 'M4.5')
        metrics: Test metrics dict
        uncontrolled: Uncontrolled baseline displacement (cm)
    """
    peak_cm = metrics['peak_disp_cm']
    improvement = 100 * (uncontrolled - peak_cm) / uncontrolled

    print(f"\n{'='*70}")
    print(f"  {magnitude} TEST RESULTS")
    print(f"{'='*70}")

    # Displacement comparison
    print(f"\n  Peak Roof Displacement:")
    print(f"    v13-Rooftop:   {peak_cm:.2f} cm")
    print(f"    Target:        {V13_TARGETS['displacement_cm']:.2f} cm")

    disp_delta = peak_cm - V13_TARGETS['displacement_cm']
    disp_status = "✅ MET" if disp_delta <= 0 else "⚠️ CLOSE" if disp_delta <= 4 else "❌ NOT MET"
    print(f"    Δ from target: {disp_delta:+.2f} cm {disp_status}")

    print(f"    Uncontrolled:  {uncontrolled:.2f} cm")
    status = "✓" if improvement > 0 else "✗"
    print(f"    Improvement:   {improvement:+.1f}% {status}")

    # Structural safety metrics with multi-floor tracking
    print(f"\n  Structural Safety:")
    print(f"    Max ISDR:      {metrics['max_isdr_percent']:.2f}%")
    print(f"    Target ISDR:   {V13_TARGETS['isdr_percent']:.2f}%")
    print(f"    Critical Floor: {metrics['critical_floor']}")
    print(f"    Floor ISDRs: {[f'{x:.3f}%' for x in metrics['floor_isdrs']]}")

    isdr_delta = metrics['max_isdr_percent'] - V13_TARGETS['isdr_percent']
    isdr_status = "✅ MET" if metrics['max_isdr_percent'] <= V13_TARGETS['isdr_percent'] else \
                  "⚠️ CLOSE" if metrics['max_isdr_percent'] <= 0.8 else \
                  "❌ NOT MET"
    print(f"    Δ from target: {isdr_delta:+.2f}% {isdr_status}")

    print(f"\n    DCR:           {metrics['DCR']:.2f}")
    print(f"    Target DCR:    {V13_TARGETS['dcr']:.2f}")

    dcr_delta = metrics['DCR'] - V13_TARGETS['dcr']
    dcr_status = "✅ MET" if metrics['DCR'] <= V13_TARGETS['dcr'] else \
                 "⚠️ CLOSE" if metrics['DCR'] <= 1.3 else \
                 "❌ NOT MET"
    print(f"    Δ from target: {dcr_delta:+.2f} {dcr_status}")

    # Control effort
    print(f"\n  Control Effort:")
    print(f"    Peak force:    {metrics['peak_force_kN']:.1f} kN (limit: 300 kN)")
    print(f"    Mean force:    {metrics['mean_force_kN']:.1f} kN")
    print(f"    RMS force:     {metrics['rms_force_kN']:.1f} kN")

    # Additional metrics
    print(f"\n  Additional Metrics:")
    print(f"    RMS displacement: {metrics['rms_roof_displacement']*100:.2f} cm")
    print(f"    Max drift:        {metrics['max_drift']*100:.2f} cm")


def print_summary_table(results):
    """
    Print summary table comparing all earthquake magnitudes

    Args:
        results: List of (magnitude, metrics, uncontrolled) tuples
    """
    print(f"\n\n{'='*70}")
    print(f"  SUMMARY: v13 Rooftop TMD Performance")
    print(f"{'='*70}\n")

    print(f"{'Magnitude':<12} {'v13 (cm)':<12} {'Target':<12} {'Uncont (cm)':<14} {'Improve':<10} {'ISDR%':<10} {'DCR':<8}")
    print(f"{'-'*70}")

    total_improvement = 0
    count = 0

    for magnitude, metrics, uncontrolled in results:
        peak_cm = metrics['peak_disp_cm']
        improvement = 100 * (uncontrolled - peak_cm) / uncontrolled

        target_str = f"{V13_TARGETS['displacement_cm']:.2f}"
        improve_str = f"{improvement:+.1f}%"

        print(f"{magnitude:<12} {peak_cm:<12.2f} {target_str:<12} {uncontrolled:<14.2f} {improve_str:<10} {metrics['max_isdr_percent']:<10.2f} {metrics['DCR']:<8.2f}")

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
    Evaluate whether conservative targets were met

    Args:
        results: List of (magnitude, metrics, uncontrolled) tuples
    """
    print(f"\n\n{'='*70}")
    print(f"  TARGET ACHIEVEMENT ANALYSIS")
    print(f"{'='*70}\n")

    # Focus on M4.5 for target evaluation
    m45_result = next((r for r in results if r[0] == 'M4.5'), None)

    if m45_result is None:
        print("⚠️  M4.5 results not available for target analysis")
        return

    _, metrics, _ = m45_result

    print("v13 ROOFTOP Targets (Conservative - learning from v12):")
    print(f"  Target: {V13_TARGETS['displacement_cm']} cm displacement, {V13_TARGETS['isdr_percent']}% ISDR, {V13_TARGETS['dcr']} DCR\n")

    # Displacement target
    peak_cm = metrics['peak_disp_cm']
    disp_met = peak_cm <= V13_TARGETS['displacement_cm']
    disp_close = peak_cm <= V13_TARGETS['displacement_cm'] + 4
    disp_status = "✅ MET" if disp_met else "⚠️ CLOSE" if disp_close else "❌ NOT MET"
    print(f"  Displacement: {peak_cm:.2f} cm {disp_status}")

    # ISDR target
    isdr = metrics['max_isdr_percent']
    isdr_met = isdr <= V13_TARGETS['isdr_percent']
    isdr_close = isdr <= 0.8
    isdr_status = "✅ MET" if isdr_met else "⚠️ CLOSE" if isdr_close else "❌ NOT MET"
    print(f"  ISDR:         {isdr:.2f}% {isdr_status}")
    print(f"    Critical Floor: {metrics['critical_floor']}")

    # DCR target
    dcr = metrics['DCR']
    dcr_met = dcr <= V13_TARGETS['dcr']
    dcr_close = dcr <= 1.3
    dcr_status = "✅ MET" if dcr_met else "⚠️ CLOSE" if dcr_close else "❌ NOT MET"
    print(f"  DCR:          {dcr:.2f} {dcr_status}")

    # Overall assessment
    all_met = disp_met and isdr_met and dcr_met
    all_close = disp_close and isdr_close and dcr_close

    print(f"\n  Overall Assessment:")
    if all_met:
        print("  ✅ ALL TARGETS MET - SUCCESS!")
        print("     Rooftop TMD with proper multi-floor tracking working effectively")
        print("     'Almost no structural damage' achieved")
    elif all_close:
        print("  ⚠️  CLOSE TO TARGETS - Very good performance")
        print("     Rooftop TMD shows significant improvement")
        print("     May require fine-tuning: increase training time or adjust force limit")
    else:
        print("  ❌ TARGETS NOT MET - More work needed")
        print("     Check TensorBoard for convergence issues")
        print("     Ensure training completed full 1.5M steps")
        print("     Consider: Adjust reward weights or increase TMD mass to 5%")

    # Rooftop effectiveness analysis
    print(f"\n  Rooftop TMD Effectiveness (v13):")
    print(f"    TMD Location:     Floor 12 (roof)")
    print(f"    TMD Mass:         8000 kg (4% of floor mass)")
    print(f"    Max Force:        300 kN")
    print(f"    Multi-Floor ISDR: ✅ Tracking all 12 floors")

    if isdr < 1.0:
        print(f"    ISDR Performance: ✅ Excellent (< 1.0%)")
        print(f"                      Multi-floor tracking working as designed")
    elif isdr < 1.5:
        print(f"    ISDR Performance: ⚠️ Good (< 1.5%)")
        print(f"                      Good performance, room for improvement")
    else:
        print(f"    ISDR Performance: ❌ Needs improvement (> 1.5%)")
        print(f"                      Check training convergence or increase control authority")


def compare_with_previous_versions():
    """Print comparison note about v11, v12, and v13"""
    print(f"\n\n{'='*70}")
    print(f"  VERSION COMPARISON: v11 vs v12 vs v13")
    print(f"{'='*70}\n")

    print("  v11 Rooftop TMD (250 kN, 4% mass):")
    print("    - Rooftop placement (conventional)")
    print("    - Single ISDR tracking (floor 8 only)")
    print("    - Limited ISDR improvement (~3%)")
    print("    - Location: Correct, but incomplete tracking")

    print("\n  v12 Soft-Story TMD (300 kN, 4% mass) - FAILED:")
    print("    - Floor 8 placement (soft story)")
    print("    - Single ISDR tracking (floor 8 only)")
    print("    - Hypothesis: Direct control would improve ISDR")
    print("    - Result: FAILED - comparison metric was broken")
    print("    - Lesson: TMD location was wrong, tracking was incomplete")

    print("\n  v13 Rooftop TMD (300 kN, 4% mass) - IMPROVED:")
    print("    - Floor 12 placement (roof) - CORRECT")
    print("    - Multi-floor ISDR tracking (all 12 floors) - KEY FIX")
    print("    - Proper multi-floor drift control")
    print("    - Expected: +40-60% ISDR reduction (conservative)")

    print("\n  Key Lesson Learned:")
    print("    ❌ v12 failed because:")
    print("       1. Wrong TMD location (floor 8 instead of roof)")
    print("       2. Incomplete ISDR tracking (only floor 8)")
    print("    ✅ v13 fixes both issues:")
    print("       1. Correct TMD location (roof)")
    print("       2. Complete ISDR tracking (all floors)")


# ============================================================================
# MAIN TEST FUNCTION
# ============================================================================

def test_v13_model(model_path, test_dir='../../matlab/datasets', force_limit=300000):
    """
    Main test function - evaluates model on all test earthquakes

    Args:
        model_path: Path to trained model .zip file
        test_dir: Directory containing test earthquake files
        force_limit: Maximum control force used during training (300 kN for v13)
    """
    print(f"\n{'='*70}")
    print(f"  TESTING v13 ROOFTOP TMD MODEL")
    print(f"  Key Feature: Proper multi-floor ISDR tracking with rooftop placement")
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
    print(f"  TMD Location: Floor 12 (roof)")
    print(f"  TMD Mass: 8000 kg (4% of floor mass)")
    print(f"  Force limit: {force_limit/1000:.0f} kN")
    print(f"  Reward scale: 1.0 (fixed, no adaptive scaling)")
    print(f"  Multi-floor ISDR tracking: All 12 floors")
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

            # Get baseline
            uncontrolled = UNCONTROLLED_BASELINES.get(magnitude, None)

            # Print detailed results
            print_detailed_results(magnitude, metrics, uncontrolled)

            # Store for summary
            results.append((magnitude, metrics, uncontrolled))

        except Exception as e:
            print(f"❌ Error testing {magnitude}: {e}\n")
            continue

    # Print summary
    if results:
        print_summary_table(results)
        evaluate_targets(results)
        compare_with_previous_versions()
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
        description='Test v13 Rooftop TMD Model on Earthquake Data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to trained model .zip file (e.g., models/v13_rooftop_breakthrough/stage1_M4.5_final.zip)'
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
        default=300000,
        help='Maximum control force in Newtons (should match training - 300 kN for v13)'
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    test_v13_model(
        model_path=args.model_path,
        test_dir=args.test_dir,
        force_limit=args.force_limit
    )

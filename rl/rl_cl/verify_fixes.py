#!/usr/bin/env python3
"""
VERIFY ALL CRITICAL FIXES APPLIED
==================================

This script verifies that all three critical bugs have been fixed:
1. Training duration limit removed
2. Observation space clipping fixed
3. Baseline drift corrected

Run this before retraining to ensure everything is ready.

Usage:
    python verify_fixes.py

Author: Siddharth
Date: December 30, 2025
"""

import sys
import os
import numpy as np

def verify_training_duration_limit():
    """Verify that max_steps limit has been removed"""
    print("\n" + "="*70)
    print("TEST 1: Training Duration Limit")
    print("="*70)

    # Check training environment
    with open('tmd_environment.py', 'r') as f:
        content = f.read()

    if 'min(len(earthquake_data), 2000)' in content:
        print("❌ FAILED: Training environment still has 2000-step limit!")
        return False

    if 'self.max_steps = len(earthquake_data)' in content:
        print("✅ PASSED: Training environment uses full duration")
    else:
        print("⚠️  WARNING: Could not verify max_steps assignment")
        return False

    # Check API environment
    api_path = '../../restapi/rl_cl/rl_cl_tmd_environment.py'
    if os.path.exists(api_path):
        with open(api_path, 'r') as f:
            api_content = f.read()

        if 'min(len(earthquake_data), 2000)' in api_content:
            print("❌ FAILED: API environment still has 2000-step limit!")
            return False

        if 'self.max_steps = len(earthquake_data)' in api_content:
            print("✅ PASSED: API environment uses full duration")
        else:
            print("⚠️  WARNING: Could not verify API max_steps")
            return False

    return True


def verify_observation_clipping():
    """Verify that RLCLController uses correct 8-value observation bounds"""
    print("\n" + "="*70)
    print("TEST 2: Observation Space Clipping")
    print("="*70)

    controller_path = '../../restapi/rl_cl/RLCLController.py'
    if not os.path.exists(controller_path):
        print("⚠️  WARNING: RLCLController.py not found")
        return False

    with open(controller_path, 'r') as f:
        content = f.read()

    # Check for 8-value bounds array
    if 'obs_bounds_array' not in content:
        print("❌ FAILED: obs_bounds_array not found in RLCLController")
        return False

    if '[-1.2, -3.0, -1.2, -3.0, -1.2, -3.0, -1.5, -3.5]' in content:
        print("✅ PASSED: 8-value observation bounds defined correctly")
    else:
        print("❌ FAILED: 8-value bounds not correctly defined")
        return False

    # Check that simulate_episode uses new bounds
    if 'obs_clipped = np.clip(obs, self.obs_bounds_array[0], self.obs_bounds_array[1])' in content:
        print("✅ PASSED: simulate_episode uses correct clipping")
    else:
        print("❌ FAILED: simulate_episode still uses old 4-value clipping")
        return False

    return True


def verify_baseline_correction():
    """Verify that earthquake datasets have zero final acceleration"""
    print("\n" + "="*70)
    print("TEST 3: Baseline Drift Correction")
    print("="*70)

    # Check a few test datasets
    test_files = [
        '../../matlab/datasets/PEER_high_M7.4_PGA0.75g.csv',
        '../../matlab/datasets/PEER_small_M4.5_PGA0.25g.csv',
    ]

    all_passed = True
    for test_file in test_files:
        if not os.path.exists(test_file):
            print(f"⚠️  WARNING: {os.path.basename(test_file)} not found")
            continue

        try:
            data = np.loadtxt(test_file, delimiter=',', skiprows=1)
            final_accel = data[-1, 1]

            if abs(final_accel) < 1e-6:
                print(f"✅ PASSED: {os.path.basename(test_file)} - final accel = {final_accel:.10f}")
            else:
                print(f"❌ FAILED: {os.path.basename(test_file)} - final accel = {final_accel:.6f} (NOT ZERO!)")
                all_passed = False
        except Exception as e:
            print(f"⚠️  ERROR reading {test_file}: {e}")
            all_passed = False

    return all_passed


def verify_observation_space_consistency():
    """Verify that training and API environments have matching observation spaces"""
    print("\n" + "="*70)
    print("TEST 4: Observation Space Consistency")
    print("="*70)

    # Check training environment observation space
    with open('tmd_environment.py', 'r') as f:
        train_content = f.read()

    if 'low=np.array([-1.2, -3.0, -1.2, -3.0, -1.2, -3.0, -1.5, -3.5])' in train_content:
        print("✅ PASSED: Training environment has 8-value observation space")
    else:
        print("❌ FAILED: Training environment observation space incorrect")
        return False

    # Check API environment observation space
    api_path = '../../restapi/rl_cl/rl_cl_tmd_environment.py'
    if os.path.exists(api_path):
        with open(api_path, 'r') as f:
            api_content = f.read()

        if 'low=np.array([-1.2, -3.0, -1.2, -3.0, -1.2, -3.0, -1.5, -3.5])' in api_content:
            print("✅ PASSED: API environment has 8-value observation space")
        else:
            print("❌ FAILED: API environment observation space incorrect")
            return False

    return True


def verify_building_parameters():
    """Verify that building parameters match MATLAB configuration"""
    print("\n" + "="*70)
    print("TEST 5: Building Parameters (MATLAB Alignment)")
    print("="*70)

    # Expected MATLAB values
    expected = {
        'floor_mass': 2.0e5,     # 200,000 kg
        'k_typical': 2.0e7,      # 20 MN/m
        'soft_factor': 0.60,     # 60%
        'damping': 0.015         # 1.5%
    }

    # Check training environment
    with open('tmd_environment.py', 'r') as f:
        train_content = f.read()

    checks = [
        ('floor_mass = 2.0e5', 'Floor mass (200,000 kg)'),
        ('k_typical = 2.0e7', 'Story stiffness (20 MN/m)'),
        ('0.60 * k_typical', 'Soft story factor (60%)'),
        ('damping_ratio = 0.015', 'Damping ratio (1.5%)')
    ]

    all_passed = True
    for check_str, desc in checks:
        if check_str in train_content:
            print(f"✅ PASSED: Training env - {desc}")
        else:
            print(f"❌ FAILED: Training env - {desc} not found")
            all_passed = False

    # Check API environment
    api_path = '../../restapi/rl_cl/rl_cl_tmd_environment.py'
    if os.path.exists(api_path):
        with open(api_path, 'r') as f:
            api_content = f.read()

        for check_str, desc in checks:
            if check_str in api_content:
                print(f"✅ PASSED: API env - {desc}")
            else:
                print(f"❌ FAILED: API env - {desc} not found")
                all_passed = False

    # Warn about old models
    if all_passed:
        print("\n⚠️  WARNING: Building parameters changed!")
        print("   Old models trained on 800 MN/m stiffness (40x stiffer)")
        print("   New models train on 20 MN/m stiffness (matches MATLAB)")
        print("   YOU MUST DELETE OLD MODELS AND RETRAIN!")

    return all_passed


def main():
    """Run all verification tests"""
    print("\n" + "="*70)
    print("VERIFYING CRITICAL FIXES")
    print("="*70)
    print("\nThis script verifies that all critical bugs have been fixed:")
    print("  1. Training duration limit removed (40s → full duration)")
    print("  2. Observation clipping fixed (4 values → 8 values)")
    print("  3. Baseline drift corrected (final accel = 0.0)")
    print("  4. Observation spaces consistent across train/API/controller")
    print("  5. Building parameters match MATLAB (20 MN/m stiffness)")

    # Run all tests
    results = []
    results.append(("Training Duration Limit", verify_training_duration_limit()))
    results.append(("Observation Clipping", verify_observation_clipping()))
    results.append(("Baseline Correction", verify_baseline_correction()))
    results.append(("Observation Space Consistency", verify_observation_space_consistency()))
    results.append(("Building Parameters (MATLAB Match)", verify_building_parameters()))

    # Summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)

    all_passed = True
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {test_name}")
        if not passed:
            all_passed = False

    print("\n" + "="*70)
    if all_passed:
        print("✅ ALL TESTS PASSED - READY FOR RETRAINING!")
        print("="*70)
        print("\nNext steps:")
        print("  1. Delete old models: rm -rf rl_cl_robust_models/*")
        print("  2. Start training: python train_final_robust_rl_cl.py")
        print("  3. Monitor for DCR improvements and stable displacements")
        return 0
    else:
        print("❌ SOME TESTS FAILED - FIX ISSUES BEFORE RETRAINING")
        print("="*70)
        return 1


if __name__ == "__main__":
    sys.exit(main())

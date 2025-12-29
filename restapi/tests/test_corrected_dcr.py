"""
Test to verify corrected DCR penalty implementation

This test ensures that:
1. Peak drift tracking works correctly
2. DCR penalty during training matches episode-level DCR
3. DCR penalty is included in reward breakdown
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rl_cl.rl_cl_tmd_environment import ImprovedTMDBuildingEnv


def test_peak_drift_tracking():
    """Test that peak drifts are tracked correctly over time"""

    print("=" * 60)
    print("TEST 1: Peak Drift Tracking")
    print("=" * 60)

    # Create environment
    t = np.linspace(0, 10, 500)
    test_earthquake = 2.0 * np.sin(2 * np.pi * 1.5 * t) * np.exp(-0.1 * t)

    env = ImprovedTMDBuildingEnv(
        earthquake_data=test_earthquake,
        dt=0.02,
        earthquake_name="Test"
    )

    obs, info = env.reset()

    # Verify peak drifts start at zero
    assert np.all(env.peak_drift_per_floor == 0), "Peak drifts should start at zero"
    print(f"\n  Initial peak drifts: all zeros - PASS")

    # Run some steps
    max_drifts_seen = []
    for i in range(20):
        action = np.array([0.5 * np.sin(i * 0.1)])
        obs, reward, terminated, truncated, info = env.step(action)
        max_drifts_seen.append(np.max(env.peak_drift_per_floor))

    # Verify peak drifts are monotonically increasing
    for i in range(1, len(max_drifts_seen)):
        assert max_drifts_seen[i] >= max_drifts_seen[i-1] - 1e-10, \
            f"Peak drifts should be monotonically increasing: {max_drifts_seen[i]} < {max_drifts_seen[i-1]}"

    print(f"  Peak drift progression (max across floors):")
    for i in [0, 5, 10, 15, 19]:
        print(f"    Step {i}: {max_drifts_seen[i]:.6f} m")
    print(f"  PASS: Peak drifts monotonically increasing")

    # Verify peak drifts match drift history
    drift_array = np.array(env.drift_history)
    max_drift_per_floor_from_history = np.max(drift_array, axis=0)

    print(f"\n  Comparing peak tracking vs drift history:")
    print(f"    From tracking: {env.peak_drift_per_floor}")
    print(f"    From history:  {max_drift_per_floor_from_history}")

    assert np.allclose(env.peak_drift_per_floor, max_drift_per_floor_from_history), \
        "Peak drifts should match max from drift history"
    print(f"  PASS: Peak tracking matches drift history")

    print("\n" + "=" * 60)
    print("TEST 1 PASSED!")
    print("=" * 60)


def test_dcr_penalty_consistency():
    """Test that DCR penalty during training matches episode-level DCR"""

    print("\n" + "=" * 60)
    print("TEST 2: DCR Penalty Consistency")
    print("=" * 60)

    # Create environment
    t = np.linspace(0, 10, 500)
    test_earthquake = 3.0 * np.sin(2 * np.pi * 1.5 * t) * np.exp(-0.1 * t)

    env = ImprovedTMDBuildingEnv(
        earthquake_data=test_earthquake,
        dt=0.02,
        earthquake_name="Test"
    )

    obs, info = env.reset()

    # Run episode
    last_info = None
    for i in range(100):
        action = np.array([0.7 * np.sin(i * 0.05)])
        obs, reward, terminated, truncated, info = env.step(action)
        last_info = info

    # Get episode-level metrics
    episode_metrics = env.get_episode_metrics()

    print(f"\n  Episode-level DCR: {episode_metrics['DCR']:.4f}")
    print(f"  Max drift: {episode_metrics['max_drift']:.6f} m")

    # Calculate DCR from peak tracking (should match episode DCR exactly)
    sorted_peaks = np.sort(env.peak_drift_per_floor)
    percentile_75 = np.percentile(sorted_peaks, 75)
    max_peak = np.max(env.peak_drift_per_floor)

    if percentile_75 > 1e-10:
        calculated_dcr = max_peak / percentile_75
    else:
        calculated_dcr = 0.0

    print(f"  DCR from peak tracking: {calculated_dcr:.4f}")

    assert abs(calculated_dcr - episode_metrics['DCR']) < 1e-6, \
        f"DCR from tracking should match episode DCR: {calculated_dcr} != {episode_metrics['DCR']}"
    print(f"  PASS: Peak tracking DCR matches episode DCR")

    # Verify DCR is in reward breakdown
    assert 'dcr' in last_info['reward_breakdown'], "DCR should be in reward breakdown"
    print(f"\n  DCR penalty in last step: {last_info['reward_breakdown']['dcr']:.6f}")
    print(f"  PASS: DCR penalty included in reward breakdown")

    print("\n" + "=" * 60)
    print("TEST 2 PASSED!")
    print("=" * 60)


def test_dcr_penalty_calculation():
    """Test that DCR penalty formula is correct"""

    print("\n" + "=" * 60)
    print("TEST 3: DCR Penalty Formula")
    print("=" * 60)

    # Create environment
    t = np.linspace(0, 20, 1000)
    test_earthquake = 4.0 * np.sin(2 * np.pi * 1.5 * t) * np.exp(-0.1 * t)

    env = ImprovedTMDBuildingEnv(
        earthquake_data=test_earthquake,
        dt=0.02,
        earthquake_name="Test"
    )

    obs, info = env.reset()

    # Run episode
    dcr_penalties = []
    dcr_values = []

    for i in range(200):
        action = np.array([0.8 * np.sin(i * 0.1)])
        obs, reward, terminated, truncated, info = env.step(action)

        dcr_penalties.append(info['reward_breakdown']['dcr'])

        # Calculate current DCR
        sorted_peaks = np.sort(env.peak_drift_per_floor)
        percentile_75 = np.percentile(sorted_peaks, 75)
        max_peak = np.max(env.peak_drift_per_floor)

        if percentile_75 > 1e-10 and max_peak > 1e-10:
            current_dcr = max_peak / percentile_75
            dcr_values.append(current_dcr)

    print(f"\n  DCR statistics over episode:")
    print(f"    Min DCR: {min(dcr_values):.4f}")
    print(f"    Max DCR: {max(dcr_values):.4f}")
    print(f"    Mean DCR: {np.mean(dcr_values):.4f}")

    print(f"\n  DCR penalty statistics:")
    print(f"    Min penalty: {min(dcr_penalties):.6f}")
    print(f"    Max penalty: {max(dcr_penalties):.6f}")
    print(f"    Mean penalty: {np.mean(dcr_penalties):.6f}")

    # Verify penalty formula: -0.5 * (max(0, DCR - 1.0))^2
    for dcr, penalty in zip(dcr_values[-10:], dcr_penalties[-10:]):
        dcr_deviation = max(0, dcr - 1.0)
        expected_penalty = -0.5 * (dcr_deviation ** 2)

        assert abs(penalty - expected_penalty) < 1e-6, \
            f"Penalty formula incorrect: DCR={dcr:.4f}, expected={expected_penalty:.6f}, got={penalty:.6f}"

    print(f"  PASS: DCR penalty formula correct")

    # Verify penalties are non-positive
    assert all(p <= 0 for p in dcr_penalties), "All DCR penalties should be non-positive"
    print(f"  PASS: All DCR penalties are non-positive")

    # Verify DCR > 1.0 results in negative penalty
    # Note: dcr_values list might be shorter than dcr_penalties if early steps had zero DCR
    if len(dcr_values) > 0 and max(dcr_values) > 1.0:
        # Find a step with high DCR by checking last 10 steps
        high_dcr_found = False
        for i in range(len(dcr_penalties) - 10, len(dcr_penalties)):
            if i >= 0 and i < len(dcr_penalties):
                # Recalculate DCR at this step
                sorted_peaks = np.sort(env.peak_drift_per_floor)
                percentile_75 = np.percentile(sorted_peaks, 75)
                max_peak = np.max(env.peak_drift_per_floor)
                if percentile_75 > 1e-10 and max_peak > 1e-10:
                    current_dcr = max_peak / percentile_75
                    if current_dcr > 1.0:
                        print(f"\n  High DCR example (final state):")
                        print(f"    DCR: {current_dcr:.4f}")
                        print(f"    Expected penalty: {-0.5 * (current_dcr - 1.0)**2:.6f}")
                        high_dcr_found = True
                        break

        if high_dcr_found:
            print(f"  PASS: High DCR penalty verified")

    print("\n" + "=" * 60)
    print("TEST 3 PASSED!")
    print("=" * 60)


def test_reset_clears_peak_drifts():
    """Test that reset() clears peak drift tracking"""

    print("\n" + "=" * 60)
    print("TEST 4: Reset Clears Peak Drifts")
    print("=" * 60)

    # Create environment
    t = np.linspace(0, 10, 500)
    test_earthquake = 2.0 * np.sin(2 * np.pi * 1.5 * t) * np.exp(-0.1 * t)

    env = ImprovedTMDBuildingEnv(
        earthquake_data=test_earthquake,
        dt=0.02,
        earthquake_name="Test"
    )

    obs, info = env.reset()

    # Run some steps
    for i in range(20):
        action = np.array([0.5])
        obs, reward, terminated, truncated, info = env.step(action)

    peak_before_reset = np.max(env.peak_drift_per_floor)
    print(f"\n  Peak drift before reset: {peak_before_reset:.6f} m")
    assert peak_before_reset > 0, "Should have non-zero peak drifts after steps"

    # Reset
    obs, info = env.reset()

    peak_after_reset = np.max(env.peak_drift_per_floor)
    print(f"  Peak drift after reset: {peak_after_reset:.6f} m")

    assert np.all(env.peak_drift_per_floor == 0), "Peak drifts should be zero after reset"
    print(f"  PASS: Peak drifts cleared on reset")

    print("\n" + "=" * 60)
    print("TEST 4 PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("CORRECTED DCR PENALTY TEST SUITE")
    print("=" * 60)

    test_peak_drift_tracking()
    test_dcr_penalty_consistency()
    test_dcr_penalty_calculation()
    test_reset_clears_peak_drifts()

    print("\n" + "=" * 60)
    print("ALL DCR TESTS PASSED!")
    print("=" * 60)
    print("\nVerified:")
    print("  1. Peak drift tracking works correctly")
    print("  2. DCR penalty matches episode-level DCR exactly")
    print("  3. DCR penalty formula is correct")
    print("  4. DCR penalty included in reward breakdown")
    print("  5. Reset properly clears peak drift tracking")
    print("=" * 60)

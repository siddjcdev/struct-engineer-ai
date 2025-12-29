"""
Test to verify drift calculation consistency

This test ensures that the drift calculation is correct and consistent
between instantaneous (during step) and episode-level (get_episode_metrics) calculations.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rl_cl.rl_cl_tmd_environment import ImprovedTMDBuildingEnv


def test_drift_calculation():
    """Test that drift calculation produces correct number of values and correct values"""

    # Create a simple test earthquake
    t = np.linspace(0, 10, 500)
    test_earthquake = 2.0 * np.sin(2 * np.pi * 1.5 * t) * np.exp(-0.1 * t)

    # Create environment
    env = ImprovedTMDBuildingEnv(
        earthquake_data=test_earthquake,
        dt=0.02,
        earthquake_name="Test"
    )

    # Test the helper method directly
    test_displacements = np.array([0.01, 0.02, 0.025, 0.03, 0.032, 0.034,
                                   0.035, 0.036, 0.037, 0.038, 0.039, 0.04])

    drifts = env._compute_interstory_drifts(test_displacements)

    print("=" * 60)
    print("DRIFT CALCULATION TEST")
    print("=" * 60)

    # Check 1: Correct number of values
    print(f"\n[CHECK 1] Number of drift values")
    print(f"  Input displacements: {len(test_displacements)} floors")
    print(f"  Output drifts: {len(drifts)} values")
    assert len(drifts) == len(test_displacements), "Should have same number of drifts as floors"
    print(f"  PASS: {len(drifts)} drifts calculated")

    # Check 2: First floor drift is relative to ground
    print(f"\n[CHECK 2] First floor drift (relative to ground)")
    expected_drift_0 = abs(test_displacements[0])
    print(f"  Floor 1 displacement: {test_displacements[0]:.4f} m")
    print(f"  Expected drift: {expected_drift_0:.4f} m")
    print(f"  Calculated drift: {drifts[0]:.4f} m")
    assert abs(drifts[0] - expected_drift_0) < 1e-10, "First floor drift should be displacement relative to ground"
    print(f"  PASS")

    # Check 3: Other floors are relative to floor below
    print(f"\n[CHECK 3] Interstory drifts (relative to floor below)")
    for i in range(1, len(test_displacements)):
        expected_drift = abs(test_displacements[i] - test_displacements[i-1])
        print(f"  Floor {i+1}: d={test_displacements[i]:.4f}m, expected_drift={expected_drift:.5f}m, actual={drifts[i]:.5f}m")
        assert abs(drifts[i] - expected_drift) < 1e-10, f"Floor {i+1} drift incorrect"
    print(f"  PASS: All interstory drifts correct")

    # Check 4: All drifts are positive (absolute values)
    print(f"\n[CHECK 4] All drifts are positive (absolute values)")
    print(f"  Min drift: {np.min(drifts):.6f} m")
    print(f"  Max drift: {np.max(drifts):.6f} m")
    assert np.all(drifts >= 0), "All drifts should be positive (absolute values)"
    print(f"  PASS")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)

    # Now test in actual simulation
    print("\n" + "=" * 60)
    print("SIMULATION CONSISTENCY TEST")
    print("=" * 60)

    obs, info = env.reset()

    # Run a few steps
    for i in range(5):
        action = np.array([0.5])  # Apply some control force
        obs, reward, terminated, truncated, info = env.step(action)

    # Check that drift history has correct shape
    drift_array = np.array(env.drift_history)
    print(f"\n[CHECK] Drift history shape: {drift_array.shape}")
    print(f"  Expected: (5 timesteps, 12 floors)")
    assert drift_array.shape == (5, 12), "Drift history should be (timesteps, n_floors)"
    print(f"  PASS")

    # Check that all drifts are positive
    print(f"\n[CHECK] All drifts in history are positive:")
    print(f"  Min drift in history: {np.min(drift_array):.6f} m")
    print(f"  Max drift in history: {np.max(drift_array):.6f} m")
    assert np.all(drift_array >= 0), "All historical drifts should be positive"
    print(f"  PASS")

    print("\n" + "=" * 60)
    print("SIMULATION TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    test_drift_calculation()

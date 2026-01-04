"""
Test script to diagnose M7.4 earthquake extreme displacement issue

This script tests the M7.4 PGA 0.75g earthquake to understand why it causes:
- Extreme displacements (1218 cm reported)
- Almost unreducible DCR with both RL models
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rl_cl.rl_cl_tmd_environment import ImprovedTMDBuildingEnv


def test_m74_earthquake():
    """Test M7.4 earthquake with no control"""

    print("=" * 70)
    print("M7.4 EARTHQUAKE ANALYSIS - NO CONTROL")
    print("=" * 70)

    # Load earthquake
    eq_file = r"c:\Dev\dAmpIng26\git\struct-engineer-ai\matlab\datasets\training_set\TRAIN_M7.4_PGA0.75g_variant1.csv"
    data = np.loadtxt(eq_file, delimiter=',', skiprows=1)

    times = data[:, 0]
    accelerations = data[:, 1]
    dt = np.mean(np.diff(times))

    print(f"\nEarthquake characteristics:")
    print(f"  Duration: {times[-1]:.2f} seconds")
    print(f"  Samples: {len(accelerations)}")
    print(f"  Time step: {dt:.4f} s")
    print(f"  PGA: {np.max(np.abs(accelerations)):.4f} m/s² ({np.max(np.abs(accelerations))/9.81:.4f}g)")

    # Create environment
    env = ImprovedTMDBuildingEnv(
        earthquake_data=accelerations,
        dt=dt,
        max_force=150000.0,
        earthquake_name="M7.4_PGA0.75g"
    )

    print(f"\nBuilding parameters:")
    print(f"  Floors: {env.n_floors}")
    print(f"  Floor mass: {env.floor_mass/1000:.1f} tonnes")
    print(f"  TMD mass: {env.tmd_mass/1000:.1f} tonnes")
    print(f"  Damping ratio: {env.damping_ratio*100:.1f}%")
    print(f"  Story stiffness: {env.story_stiffness[0]/1e6:.0f} MN/m (typical)")
    print(f"  Soft story (floor 8): {env.story_stiffness[7]/1e6:.0f} MN/m")

    # Run with NO CONTROL
    obs, info = env.reset()

    max_disp_history = []
    max_drift_history = []
    floor_disp_history = []

    for i in range(len(accelerations)):
        action = np.array([0.0])  # NO CONTROL
        obs, reward, terminated, truncated, info = env.step(action)

        max_disp_history.append(abs(obs[0]))  # Roof displacement

        # Track all floor displacements
        floor_disp_history.append(env.displacement[:env.n_floors].copy())

        if (i+1) % 500 == 0:
            print(f"  Step {i+1}/{len(accelerations)}: Roof disp = {obs[0]*100:.2f} cm")

    # Get final metrics
    metrics = env.get_episode_metrics()

    print(f"\n" + "=" * 70)
    print("RESULTS - NO CONTROL")
    print("=" * 70)
    print(f"Peak roof displacement: {metrics['peak_roof_displacement']*100:.2f} cm")
    print(f"RMS roof displacement: {metrics['rms_roof_displacement']*100:.2f} cm")
    print(f"Max drift: {metrics['max_drift']*100:.2f} cm")
    print(f"DCR: {metrics['DCR']:.4f}")

    # Analyze drift distribution
    drift_array = np.array(env.drift_history)
    max_drift_per_floor = np.max(drift_array, axis=0)

    print(f"\nDrift distribution by floor:")
    for i, drift in enumerate(max_drift_per_floor, 1):
        star = "***" if i == 8 else ""  # Mark soft story
        print(f"  Floor {i:2d}: {drift*100:6.2f} cm {star}")

    print(f"\nDCR Analysis:")
    sorted_drifts = np.sort(max_drift_per_floor)
    p75 = np.percentile(sorted_drifts, 75)
    print(f"  75th percentile drift: {p75*100:.2f} cm")
    print(f"  Max drift: {np.max(max_drift_per_floor)*100:.2f} cm")
    print(f"  DCR = {np.max(max_drift_per_floor)/p75:.4f}")
    print(f"  Concentration at floor: {np.argmax(max_drift_per_floor)+1}")

    # Check if it's the soft story
    if np.argmax(max_drift_per_floor) == 7:  # Index 7 = Floor 8
        print(f"\n  WARNING: SOFT STORY DOMINATES - This is expected!")
        print(f"  Soft story has {env.story_stiffness[0]/env.story_stiffness[7]:.1f}x less stiffness")

    # Check final displacements
    floor_disp_array = np.array(floor_disp_history)
    max_disp_per_floor = np.max(np.abs(floor_disp_array), axis=0)

    print(f"\nMax displacement per floor:")
    for i, disp in enumerate(max_disp_per_floor, 1):
        star = "***" if i == 8 else ""
        print(f"  Floor {i:2d}: {disp*100:6.2f} cm {star}")

    return env, metrics


def test_m74_with_max_control():
    """Test M7.4 earthquake with maximum control force"""

    print("\n" + "=" * 70)
    print("M7.4 EARTHQUAKE ANALYSIS - MAXIMUM CONTROL")
    print("=" * 70)

    # Load earthquake
    eq_file = r"c:\Dev\dAmpIng26\git\struct-engineer-ai\matlab\datasets\training_set\TRAIN_M7.4_PGA0.75g_variant1.csv"
    data = np.loadtxt(eq_file, delimiter=',', skiprows=1)

    accelerations = data[:, 1]
    dt = np.mean(np.diff(data[:, 0]))

    # Create environment
    env = ImprovedTMDBuildingEnv(
        earthquake_data=accelerations,
        dt=dt,
        max_force=150000.0,
        earthquake_name="M7.4_PGA0.75g"
    )

    # Run with MAXIMUM CONTROL (trying to oppose displacement)
    obs, info = env.reset()

    for i in range(len(accelerations)):
        # Simple control: oppose roof displacement
        if obs[0] > 0:
            action = np.array([-1.0])  # Push opposite direction
        else:
            action = np.array([1.0])

        obs, reward, terminated, truncated, info = env.step(action)

    # Get final metrics
    metrics = env.get_episode_metrics()

    print(f"\nRESULTS - MAXIMUM CONTROL (simple opposing):")
    print(f"  Peak roof displacement: {metrics['peak_roof_displacement']*100:.2f} cm")
    print(f"  RMS roof displacement: {metrics['rms_roof_displacement']*100:.2f} cm")
    print(f"  Max drift: {metrics['max_drift']*100:.2f} cm")
    print(f"  DCR: {metrics['DCR']:.4f}")
    print(f"  Peak force: {metrics['peak_force_kN']:.1f} kN")

    # Compare drift distribution
    drift_array = np.array(env.drift_history)
    max_drift_per_floor = np.max(drift_array, axis=0)

    print(f"\nDrift distribution with control:")
    for i, drift in enumerate(max_drift_per_floor, 1):
        star = "***" if i == 8 else ""
        print(f"  Floor {i:2d}: {drift*100:6.2f} cm {star}")

    return env, metrics


def compare_earthquakes():
    """Compare M7.4 with other earthquakes to understand the issue"""

    print("\n" + "=" * 70)
    print("EARTHQUAKE COMPARISON")
    print("=" * 70)

    # Test several earthquakes
    dataset_dir = r"c:\Dev\dAmpIng26\git\struct-engineer-ai\matlab\datasets\training_set"

    earthquakes = [
        "TRAIN_M4.5_PGA0.15g_variant1.csv",
        "TRAIN_M5.7_PGA0.35g_variant1.csv",
        "TRAIN_M7.4_PGA0.75g_variant1.csv",
    ]

    results = []

    for eq_file in earthquakes:
        full_path = os.path.join(dataset_dir, eq_file)
        if not os.path.exists(full_path):
            print(f"  Skipping {eq_file} (not found)")
            continue

        data = np.loadtxt(full_path, delimiter=',', skiprows=1)
        accelerations = data[:, 1]
        dt = np.mean(np.diff(data[:, 0]))
        pga = np.max(np.abs(accelerations))

        # Test with no control
        env = ImprovedTMDBuildingEnv(
            earthquake_data=accelerations,
            dt=dt,
            max_force=150000.0,
            earthquake_name=eq_file
        )

        obs, info = env.reset()
        for i in range(len(accelerations)):
            action = np.array([0.0])
            obs, reward, terminated, truncated, info = env.step(action)

        metrics = env.get_episode_metrics()

        results.append({
            'name': eq_file.replace('TRAIN_', '').replace('.csv', ''),
            'pga_g': pga / 9.81,
            'peak_disp_cm': metrics['peak_roof_displacement'] * 100,
            'max_drift_cm': metrics['max_drift'] * 100,
            'dcr': metrics['DCR']
        })

    print(f"\n{'Earthquake':<30} {'PGA (g)':<10} {'Peak (cm)':<12} {'Drift (cm)':<12} {'DCR':<8}")
    print("-" * 72)
    for r in results:
        print(f"{r['name']:<30} {r['pga_g']:<10.2f} {r['peak_disp_cm']:<12.2f} {r['max_drift_cm']:<12.2f} {r['dcr']:<8.4f}")


if __name__ == "__main__":
    # Test 1: No control
    env_no_control, metrics_no_control = test_m74_earthquake()

    # Test 2: Maximum control
    env_max_control, metrics_max_control = test_m74_with_max_control()

    # Test 3: Compare with other earthquakes
    compare_earthquakes()

    print("\n" + "=" * 70)
    print("DIAGNOSIS COMPLETE")
    print("=" * 70)

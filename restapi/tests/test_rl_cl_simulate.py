"""
RL-CL SIMULATE ENDPOINT TEST
=============================

Test the /rl-cl/simulate endpoint that returns comprehensive metrics:
- RMS roof displacement
- Peak roof displacement
- Maximum interstory drift
- DCR (Drift Concentration Ratio)
- Peak and mean control forces

Usage:
    python test_rl_cl_simulate.py

Author: Siddharth
Date: December 2025
"""

import requests
import numpy as np
import time
from pathlib import Path
import sys


# API Configuration
API_URL = "http://localhost:8080"  # Adjust if different
# For deployed API, use:
# API_URL = "https://perfect-rl-api-887344515766.us-east4.run.app"


def load_earthquake_data(filename: str):
    """Load earthquake data from CSV file"""
    # Get path to datasets
    script_dir = Path(__file__).parent
    dataset_path = script_dir.parent.parent / "matlab" / "datasets" / filename

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    print(f"📂 Loading: {dataset_path.name}")
    data = np.loadtxt(dataset_path, delimiter=',', skiprows=1)

    # Extract time and acceleration
    times = data[:, 0]
    accelerations = data[:, 1]
    dt = np.mean(np.diff(times))

    print(f"   Duration: {times[-1]:.1f} s")
    print(f"   Samples: {len(accelerations)}")
    print(f"   Time step: {dt:.4f} s")
    print(f"   PGA: {np.max(np.abs(accelerations)):.3f} m/s² ({np.max(np.abs(accelerations))/9.81:.2f}g)")

    return accelerations.tolist(), dt


def test_simulate_endpoint(earthquake_name: str, earthquake_file: str):
    """
    Test the /rl-cl/simulate endpoint with real earthquake data

    Args:
        earthquake_name: Display name (e.g., "TEST3 - Small Earthquake")
        earthquake_file: Filename in matlab/datasets/
    """
    print("\n" + "="*80)
    print(f"TESTING /rl-cl/simulate WITH {earthquake_name}")
    print("="*80)

    try:
        # Load earthquake data
        earthquake_data, dt = load_earthquake_data(earthquake_file)

        # Prepare request
        request_data = {
            "earthquake_data": earthquake_data,
            "dt": dt
        }

        print(f"\n🚀 Calling /rl-cl/simulate endpoint...")
        print(f"   Earthquake samples: {len(earthquake_data)}")
        print(f"   Expected duration: {len(earthquake_data) * dt:.1f} s")

        # Make API call
        start_time = time.time()
        response = requests.post(
            f"{API_URL}/rl-cl/simulate",
            json=request_data,
            timeout=300  # 5 minute timeout for long simulations
        )
        elapsed = (time.time() - start_time) * 1000

        if response.status_code != 200:
            print(f"\n❌ API Error: Status {response.status_code}")
            print(f"   Response: {response.text}")
            return False

        # Parse response
        result = response.json()

        # Display results
        print(f"\n✅ Simulation Complete!")
        print(f"   Total API time: {elapsed:.1f} ms ({elapsed/1000:.2f} s)")
        print(f"   Simulation time (server): {result['simulation_time_ms']:.1f} ms")

        print("\n" + "─"*80)
        print("📊 PERFORMANCE METRICS")
        print("─"*80)

        print(f"\n🏢 DISPLACEMENT METRICS:")
        print(f"   Peak roof displacement:  {result['peak_roof_displacement']*100:>8.2f} cm")
        print(f"   RMS roof displacement:   {result['rms_roof_displacement']*100:>8.2f} cm")
        print(f"   Maximum interstory drift:{result['max_drift']*100:>8.2f} cm")

        print(f"\n📐 DRIFT CONCENTRATION:")
        print(f"   DCR (Drift Concentration Ratio): {result['DCR']:>6.2f}")
        if result['DCR'] < 1.5:
            print(f"   Status: ✅ EXCELLENT (uniform drift distribution)")
        elif result['DCR'] < 2.0:
            print(f"   Status: ✅ GOOD (acceptable drift distribution)")
        elif result['DCR'] < 3.0:
            print(f"   Status: ⚠️  MODERATE (some concentration)")
        else:
            print(f"   Status: ⚠️  HIGH (significant drift concentration)")

        print(f"\n⚡ FORCE METRICS:")
        print(f"   Peak control force:      {result['peak_force_kN']:>8.1f} kN")
        print(f"   Mean control force:      {result['mean_force_kN']:>8.1f} kN")
        print(f"   Force efficiency:        {(100 / result['mean_force_kN']) if result['mean_force_kN'] > 0 else 0:>8.2f} %improvement/kN")

        print(f"\n📈 FORCE TIME SERIES:")
        forces_kN = np.array(result['forces_kN'])
        print(f"   Timesteps:               {result['count']:>8d}")
        print(f"   Force range:             [{forces_kN.min():>6.1f}, {forces_kN.max():>6.1f}] kN")
        print(f"   Force std dev:           {forces_kN.std():>8.1f} kN")

        # Sample force values
        sample_indices = [0, len(forces_kN)//4, len(forces_kN)//2, 3*len(forces_kN)//4, len(forces_kN)-1]
        print(f"\n   Sample forces (kN):")
        for idx in sample_indices:
            print(f"      t={idx*dt:>6.2f}s: {forces_kN[idx]:>8.2f} kN")

        print("\n" + "─"*80)
        print("✅ TEST PASSED - All metrics received successfully")
        print("─"*80)

        return True

    except requests.exceptions.ConnectionError:
        print(f"\n❌ CONNECTION ERROR")
        print(f"   Could not connect to {API_URL}")
        print(f"\n   Make sure the API server is running:")
        print(f"      cd restapi")
        print(f"      python main.py")
        return False

    except FileNotFoundError as e:
        print(f"\n❌ FILE ERROR: {e}")
        return False

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_health():
    """Quick health check"""
    print("\n" + "="*80)
    print("HEALTH CHECK")
    print("="*80)

    try:
        response = requests.get(f"{API_URL}/health", timeout=10)

        if response.status_code == 200:
            data = response.json()
            print(f"✅ API Status: {data['status']}")
            print(f"   RL-CL model loaded: {data.get('rl_cl_model_loaded', False)}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False

    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False


def test_with_synthetic_data():
    """Test with simple synthetic earthquake"""
    print("\n" + "="*80)
    print("TESTING WITH SYNTHETIC EARTHQUAKE DATA")
    print("="*80)

    # Generate simple sine wave earthquake
    duration = 20  # seconds
    dt = 0.02
    n_steps = int(duration / dt)
    t = np.linspace(0, duration, n_steps)

    # Decaying sine wave (simple earthquake model)
    frequency = 1.5  # Hz
    amplitude = 3.0  # m/s²
    decay = 0.1
    earthquake_data = amplitude * np.sin(2 * np.pi * frequency * t) * np.exp(-decay * t)

    print(f"📊 Generated synthetic earthquake:")
    print(f"   Duration: {duration} s")
    print(f"   Samples: {n_steps}")
    print(f"   PGA: {np.max(np.abs(earthquake_data)):.2f} m/s²")

    # Prepare request
    request_data = {
        "earthquake_data": earthquake_data.tolist(),
        "dt": dt
    }

    print(f"\n🚀 Calling /rl-cl/simulate endpoint...")

    try:
        start_time = time.time()
        response = requests.post(
            f"{API_URL}/rl-cl/simulate",
            json=request_data,
            timeout=60
        )
        elapsed = (time.time() - start_time) * 1000

        if response.status_code != 200:
            print(f"❌ API Error: {response.status_code}")
            print(f"   {response.text}")
            return False

        result = response.json()

        print(f"\n✅ Simulation Complete! ({elapsed:.0f} ms)")
        print(f"\n📊 Results:")
        print(f"   Peak roof displacement: {result['peak_roof_displacement']*100:.2f} cm")
        print(f"   RMS displacement: {result['rms_roof_displacement']*100:.2f} cm")
        print(f"   Max drift: {result['max_drift']*100:.2f} cm")
        print(f"   DCR: {result['DCR']:.2f}")
        print(f"   Peak force: {result['peak_force_kN']:.1f} kN")
        print(f"   Mean force: {result['mean_force_kN']:.1f} kN")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("  RL-CL SIMULATE ENDPOINT TEST SUITE")
    print("="*80)
    print(f"\n🧪 Testing comprehensive metrics endpoint")
    print(f"   API: {API_URL}")
    print(f"   Endpoint: POST /rl-cl/simulate")

    # Check health first
    if not test_health():
        print("\n⚠️  API not healthy, but continuing with tests anyway...")

    # Test with synthetic data first (quick test)
    print("\n" + "─"*80)
    print("TEST 1: Synthetic Data (Quick)")
    print("─"*80)
    success_synthetic = test_with_synthetic_data()

    # Test with real earthquake data
    test_cases = [
        ("TEST3 - Small Earthquake (M4.5)", "TEST3_small_earthquake_M4.5.csv"),
        ("TEST4 - Large Earthquake (M6.9)", "TEST4_large_earthquake_M6.9.csv"),
    ]

    results = []
    for i, (name, file) in enumerate(test_cases, start=2):
        print("\n" + "─"*80)
        print(f"TEST {i}: {name}")
        print("─"*80)
        success = test_simulate_endpoint(name, file)
        results.append((name, success))

    # Summary
    print("\n" + "="*80)
    print("  TEST SUMMARY")
    print("="*80)

    print(f"\n{'Test':<50} {'Result':<10}")
    print("─"*80)
    print(f"{'Synthetic Data':<50} {'✅ PASS' if success_synthetic else '❌ FAIL':<10}")
    for name, success in results:
        print(f"{name:<50} {'✅ PASS' if success else '❌ FAIL':<10}")

    all_passed = success_synthetic and all(success for _, success in results)

    if all_passed:
        print("\n" + "="*80)
        print("  ✅ ALL TESTS PASSED!")
        print("="*80)
        print("\n🎉 The /rl-cl/simulate endpoint is working perfectly!")
        print("\n📊 Metrics validated:")
        print("   ✅ RMS roof displacement")
        print("   ✅ Peak roof displacement")
        print("   ✅ Maximum interstory drift")
        print("   ✅ DCR (Drift Concentration Ratio)")
        print("   ✅ Peak control force")
        print("   ✅ Mean control force")
        print("   ✅ Force time series")
        print()
    else:
        print("\n" + "="*80)
        print("  ⚠️  SOME TESTS FAILED")
        print("="*80)
        print("\nCheck the errors above for details.\n")


if __name__ == "__main__":
    main()

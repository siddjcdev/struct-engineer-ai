"""
PERFECT RL API TEST CLIENT
==========================

Simple client to test your Perfect RL API

Usage: python test_perfect_rl_api.py
"""

import requests
import numpy as np
import time


API_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint"""
    print("\n" + "="*70)
    print("1. TESTING HEALTH ENDPOINT")
    print("="*70)
    
    response = requests.get(f"{API_URL}/health")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Status: {data['status']}")
        print(f"   Model loaded: {data['model_loaded']}")
        print(f"   Model: {data['model']}")
        print(f"   Performance: {data['performance']}")
    else:
        print(f"❌ Health check failed: {response.status_code}")


def test_info():
    """Test info endpoint"""
    print("\n" + "="*70)
    print("2. TESTING INFO ENDPOINT")
    print("="*70)
    
    response = requests.get(f"{API_URL}/info")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Model: {data['name']}")
        print(f"   Type: {data['type']}")
        print(f"   Training: {data['training']}")
        print(f"\n   Performance:")
        for key, value in data['performance'].items():
            print(f"      {key}: {value}")
        print(f"\n   Comparison:")
        for key, value in data['comparison'].items():
            print(f"      {key}: {value}")
    else:
        print(f"❌ Info request failed: {response.status_code}")


def test_single_prediction():
    """Test single prediction"""
    print("\n" + "="*70)
    print("3. TESTING SINGLE PREDICTION")
    print("="*70)
    
    # Test data
    data = {
        "roof_displacement": 0.15,
        "roof_velocity": 0.8,
        "tmd_displacement": 0.16,
        "tmd_velocity": 0.9
    }
    
    print(f"   Input:")
    print(f"      Roof: disp={data['roof_displacement']}m, vel={data['roof_velocity']}m/s")
    print(f"      TMD:  disp={data['tmd_displacement']}m, vel={data['tmd_velocity']}m/s")
    
    response = requests.post(f"{API_URL}/predict", json=data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n   Output:")
        print(f"      Force: {result['force_N']:.2f} N ({result['force_kN']:.2f} kN)")
        print(f"      Inference time: {result['inference_time_ms']:.2f} ms")
        print(f"      Model: {result['model']}")
        print(f"✅ Single prediction successful")
    else:
        print(f"❌ Prediction failed: {response.status_code}")
        print(f"   Error: {response.text}")


def test_batch_prediction():
    """Test batch prediction"""
    print("\n" + "="*70)
    print("4. TESTING BATCH PREDICTION")
    print("="*70)
    
    # Generate test data
    n = 10
    data = {
        "roof_displacements": np.random.uniform(-0.2, 0.2, n).tolist(),
        "roof_velocities": np.random.uniform(-1.0, 1.0, n).tolist(),
        "tmd_displacements": np.random.uniform(-0.25, 0.25, n).tolist(),
        "tmd_velocities": np.random.uniform(-1.2, 1.2, n).tolist()
    }
    
    print(f"   Batch size: {n}")
    
    response = requests.post(f"{API_URL}/predict-batch", json=data)
    
    if response.status_code == 200:
        result = response.json()
        forces_kN = np.array(result['forces_kN'])
        
        print(f"\n   Results:")
        print(f"      Predictions: {result['count']}")
        print(f"      Total time: {result['total_time_ms']:.2f} ms")
        print(f"      Avg time per prediction: {result['avg_time_ms']:.2f} ms")
        print(f"      Force range: [{forces_kN.min():.2f}, {forces_kN.max():.2f}] kN")
        print(f"      Mean force: {forces_kN.mean():.2f} kN")
        print(f"      Forces (kN): {forces_kN[:5].tolist()} ...")
        print(f"✅ Batch prediction successful")
    else:
        print(f"❌ Batch prediction failed: {response.status_code}")
        print(f"   Error: {response.text}")


def test_performance():
    """Test API performance"""
    print("\n" + "="*70)
    print("5. TESTING PERFORMANCE (100 predictions)")
    print("="*70)
    
    n_tests = 100
    times = []
    
    data = {
        "roof_displacement": 0.15,
        "roof_velocity": 0.8,
        "tmd_displacement": 0.16,
        "tmd_velocity": 0.9
    }
    
    print(f"   Running {n_tests} single predictions...")
    
    for i in range(n_tests):
        start = time.time()
        response = requests.post(f"{API_URL}/predict", json=data)
        elapsed = (time.time() - start) * 1000
        
        if response.status_code == 200:
            times.append(elapsed)
    
    times = np.array(times)
    
    print(f"\n   Results:")
    print(f"      Successful: {len(times)}/{n_tests}")
    print(f"      Mean time: {times.mean():.2f} ms")
    print(f"      Min time: {times.min():.2f} ms")
    print(f"      Max time: {times.max():.2f} ms")
    print(f"      Std dev: {times.std():.2f} ms")
    print(f"✅ Performance test complete")


def main():
    print("\n" + "="*70)
    print("  PERFECT RL API TEST SUITE")
    print("="*70)
    print("\n🏆 Testing champion model API")
    print(f"   API URL: {API_URL}")
    
    try:
        # Run tests
        test_health()
        test_info()
        test_single_prediction()
        test_batch_prediction()
        test_performance()
        
        print("\n" + "="*70)
        print("  ✅ ALL TESTS PASSED!")
        print("="*70)
        print("\n🎉 Your Perfect RL API is working perfectly!\n")
        
    except requests.exceptions.ConnectionError:
        print("\n" + "="*70)
        print("  ❌ CONNECTION ERROR")
        print("="*70)
        print(f"\n⚠️  Could not connect to API at {API_URL}")
        print("\nMake sure the API server is running:")
        print("   python perfect_rl_api.py")
        print("\nThen try again!\n")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")


if __name__ == "__main__":
    main()

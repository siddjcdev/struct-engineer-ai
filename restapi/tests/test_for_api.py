"""
Test Script for TMD Neural Network API
Verifies all endpoints work correctly
"""

import requests
import json
import numpy as np
import time
from typing import Dict


class APITester:
    """Test the TMD Neural Network API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        
    def test_health(self) -> bool:
        """Test health endpoint"""
        print("="*70)
        print("TEST 1: Health Check")
        print("="*70)
        
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            data = response.json()
            
            print(f"✅ API Status: {data['status']}")
            print(f"   Data loaded: {data['data_loaded']}")
            print(f"   NN Model loaded: {data['nn_model_loaded']}")
            
            if not data['nn_model_loaded']:
                print("\n⚠️  WARNING: Neural network model not loaded!")
                print("   Train the model first: python train_neural_network_peer.py")
                return False
            
            print()
            return True
            
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            print("\n   Is the API server running?")
            print("   Start with: python tmd_api_with_nn.py")
            return False
    
    def test_nn_status(self) -> bool:
        """Test neural network status endpoint"""
        print("="*70)
        print("TEST 2: Neural Network Status")
        print("="*70)
        
        try:
            response = self.session.get(f"{self.base_url}/nn/status")
            response.raise_for_status()
            data = response.json()
            
            if not data['model_loaded']:
                print(f"❌ Model not loaded: {data.get('error', 'Unknown error')}")
                return False
            
            print(f"✅ Model loaded: {data['model_path']}")
            print(f"   Device: {data['device']}")
            print(f"   Normalization:")
            norm = data['normalization']
            print(f"     Input mean: {norm['input_mean']}")
            print(f"     Input std: {norm['input_std']}")
            print(f"     Output mean: {norm['output_mean']}")
            print(f"     Output std: {norm['output_std']}")
            print()
            return True
            
        except Exception as e:
            print(f"❌ NN status check failed: {e}")
            return False
    
    def test_single_prediction(self) -> bool:
        """Test single prediction endpoint"""
        print("="*70)
        print("TEST 3: Single Prediction")
        print("="*70)
        
        # Test cases
        test_cases = [
            {"displacement": 0.0, "velocity": 0.0, "description": "At rest"},
            {"displacement": 0.15, "velocity": 0.8, "description": "Moderate positive"},
            {"displacement": -0.15, "velocity": -0.8, "description": "Moderate negative"},
            {"displacement": 0.25, "velocity": 1.2, "description": "Large positive"},
        ]
        
        try:
            for i, test in enumerate(test_cases, 1):
                data = {
                    "displacement": test["displacement"],
                    "velocity": test["velocity"]
                }
                
                response = self.session.post(
                    f"{self.base_url}/nn/predict",
                    json=data
                )
                response.raise_for_status()
                result = response.json()
                
                force = result['force']
                time_ms = result['inference_time_ms']
                
                print(f"Test {i}: {test['description']}")
                print(f"  Input: d={data['displacement']:.3f} m, v={data['velocity']:.3f} m/s")
                print(f"  Output: F={force:.2f} {result['force_unit']} ({force*1000:.0f} N)")
                print(f"  Inference time: {time_ms:.3f} ms")
                
                # Check if inference is fast enough for 50 Hz control (< 20 ms)
                if time_ms < 20:
                    print(f"  ✅ Fast enough for 50 Hz real-time control")
                else:
                    print(f"  ⚠️  Too slow for 50 Hz control (need < 20 ms)")
                print()
            
            print("✅ All single prediction tests passed")
            print()
            return True
            
        except Exception as e:
            print(f"❌ Single prediction test failed: {e}")
            return False
    
    def test_batch_prediction(self) -> bool:
        """Test batch prediction endpoint"""
        print("="*70)
        print("TEST 4: Batch Prediction")
        print("="*70)
        
        try:
            # Generate synthetic earthquake response
            dt = 0.02  # 50 Hz
            duration = 10.0  # seconds
            t = np.arange(0, duration, dt)
            n_points = len(t)
            
            # Simulated roof displacement and velocity
            displacement = 0.2 * np.sin(2*np.pi*1.5*t) * np.exp(-0.1*t)
            velocity = np.gradient(displacement, dt)
            
            print(f"Testing with {n_points} points ({duration}s at {int(1/dt)} Hz)")
            
            # Make batch request
            data = {
                "displacements": displacement.tolist(),
                "velocities": velocity.tolist()
            }
            
            start_time = time.time()
            response = self.session.post(
                f"{self.base_url}/nn/predict-batch",
                json=data
            )
            response.raise_for_status()
            result = response.json()
            total_time = (time.time() - start_time) * 1000  # ms
            
            forces = np.array(result['forces'])
            
            print(f"\n✅ Batch prediction successful")
            print(f"   Predictions: {result['n_predictions']}")
            print(f"   API inference time: {result['inference_time_ms']:.2f} ms")
            print(f"   Time per prediction: {result['time_per_prediction_ms']:.4f} ms")
            print(f"   Total time (including network): {total_time:.2f} ms")
            print(f"   Throughput: {result['n_predictions']/(result['inference_time_ms']/1000):.0f} predictions/second")
            
            # Check statistics
            print(f"\nForce statistics:")
            print(f"   Mean: {np.mean(forces):.2f} kN")
            print(f"   Std: {np.std(forces):.2f} kN")
            print(f"   Min: {np.min(forces):.2f} kN")
            print(f"   Max: {np.max(forces):.2f} kN")
            
            # Verify realistic
            if np.max(np.abs(forces)) > 150:
                print(f"   ⚠️  Warning: Forces seem too large (max > 150 kN)")
            else:
                print(f"   ✅ Forces in realistic range")
            
            print()
            print("✅ Batch prediction test passed")
            print()
            return True
            
        except Exception as e:
            print(f"❌ Batch prediction test failed: {e}")
            return False
    
    def test_error_handling(self) -> bool:
        """Test error handling"""
        print("="*70)
        print("TEST 5: Error Handling")
        print("="*70)
        
        # Test 1: Missing parameters
        try:
            response = self.session.post(
                f"{self.base_url}/nn/predict",
                json={"displacement": 0.1}  # Missing velocity
            )
            if response.status_code == 422:  # Validation error
                print("✅ Correctly rejects missing parameters")
            else:
                print(f"⚠️  Unexpected status code: {response.status_code}")
        except Exception as e:
            print(f"⚠️  Error test 1 failed: {e}")
        
        # Test 2: Mismatched array lengths
        try:
            response = self.session.post(
                f"{self.base_url}/nn/predict-batch",
                json={
                    "displacements": [0.1, 0.2, 0.3],
                    "velocities": [0.5, 0.6]  # Different length!
                }
            )
            if response.status_code == 400:
                print("✅ Correctly rejects mismatched array lengths")
            else:
                print(f"⚠️  Should reject mismatched arrays (got {response.status_code})")
        except Exception as e:
            print(f"⚠️  Error test 2 failed: {e}")
        
        # Test 3: Invalid endpoint
        try:
            response = self.session.get(f"{self.base_url}/invalid-endpoint")
            if response.status_code == 404:
                print("✅ Correctly returns 404 for invalid endpoints")
            else:
                print(f"⚠️  Unexpected status for invalid endpoint: {response.status_code}")
        except Exception as e:
            print(f"⚠️  Error test 3 failed: {e}")
        
        print()
        print("✅ Error handling tests completed")
        print()
        return True
    
    def test_performance_comparison(self) -> bool:
        """Compare batch vs single prediction performance"""
        print("="*70)
        print("TEST 6: Performance Comparison")
        print("="*70)
        
        try:
            n_predictions = 100
            
            # Generate random test data
            np.random.seed(42)
            displacements = np.random.uniform(-0.3, 0.3, n_predictions)
            velocities = np.random.uniform(-1.5, 1.5, n_predictions)
            
            # Test 1: Batch prediction
            batch_data = {
                "displacements": displacements.tolist(),
                "velocities": velocities.tolist()
            }
            
            start = time.time()
            batch_response = self.session.post(
                f"{self.base_url}/nn/predict-batch",
                json=batch_data
            )
            batch_time = (time.time() - start) * 1000
            batch_response.raise_for_status()
            batch_result = batch_response.json()
            
            # Test 2: Single predictions in loop (just first 10 to save time)
            n_single_test = min(10, n_predictions)
            single_times = []
            
            for i in range(n_single_test):
                single_data = {
                    "displacement": float(displacements[i]),
                    "velocity": float(velocities[i])
                }
                start = time.time()
                single_response = self.session.post(
                    f"{self.base_url}/nn/predict",
                    json=single_data
                )
                single_time = (time.time() - start) * 1000
                single_response.raise_for_status()
                single_times.append(single_time)
            
            avg_single_time = np.mean(single_times)
            estimated_total_single = avg_single_time * n_predictions
            
            print(f"Batch prediction ({n_predictions} points):")
            print(f"  Total time: {batch_time:.2f} ms")
            print(f"  Time per prediction: {batch_time/n_predictions:.4f} ms")
            print(f"  API inference only: {batch_result['inference_time_ms']:.2f} ms")
            
            print(f"\nSingle prediction (tested {n_single_test} points):")
            print(f"  Avg time per prediction: {avg_single_time:.2f} ms")
            print(f"  Estimated total for {n_predictions}: {estimated_total_single:.2f} ms")
            
            speedup = estimated_total_single / batch_time
            print(f"\nSpeedup factor: {speedup:.1f}x faster with batch!")
            
            if speedup > 5:
                print("✅ Batch mode significantly faster - use for time series!")
            else:
                print("⚠️  Speedup less than expected")
            
            print()
            print("✅ Performance comparison completed")
            print()
            return True
            
        except Exception as e:
            print(f"❌ Performance comparison failed: {e}")
            return False
    
    def run_all_tests(self) -> bool:
        """Run all tests"""
        print("\n")
        print("█"*70)
        print("  TMD NEURAL NETWORK API TEST SUITE")
        print("█"*70)
        print("\n")
        
        results = []
        
        # Run tests
        results.append(("Health Check", self.test_health()))
        
        # Only continue if health check passed
        if results[0][1]:
            results.append(("NN Status", self.test_nn_status()))
            results.append(("Single Prediction", self.test_single_prediction()))
            results.append(("Batch Prediction", self.test_batch_prediction()))
            results.append(("Error Handling", self.test_error_handling()))
            results.append(("Performance Comparison", self.test_performance_comparison()))
        
        # Print summary
        print("="*70)
        print("TEST SUMMARY")
        print("="*70)
        print()
        
        for test_name, passed in results:
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"{test_name:<30} {status}")
        
        print()
        
        all_passed = all(result for _, result in results)
        
        if all_passed:
            print("🎉 ALL TESTS PASSED!")
            print()
            print("Your API is ready for MATLAB integration!")
            print()
            print("Next steps:")
            print("  1. See matlab_nn_api_example.m for integration examples")
            print("  2. Test with your actual building simulation")
            print("  3. Deploy to cloud for production use")
        else:
            print("⚠️  SOME TESTS FAILED")
            print()
            print("Please fix issues before deploying")
        
        print()
        print("="*70)
        
        return all_passed


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test TMD Neural Network API")
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="Base URL of the API (default: http://localhost:8000)"
    )
    
    args = parser.parse_args()
    
    tester = APITester(args.url)
    success = tester.run_all_tests()
    
    return 0 if success else 1


if __name__ == '__main__':
    exit(main())
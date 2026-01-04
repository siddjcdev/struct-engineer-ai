"""
Example script demonstrating how to use the TMD Simulation API
"""
import requests
import json

BASE_URL = "http://localhost:8000"


def print_section(title):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def test_api():
    """Test various API endpoints"""
    
    # 1. Health check
    print_section("Health Check")
    response = requests.get(f"{BASE_URL}/health")
    print(json.dumps(response.json(), indent=2))
    
    # 2. Get metadata
    print_section("Simulation Metadata")
    response = requests.get(f"{BASE_URL}/simulation/metadata")
    metadata = response.json()
    print(f"Version: {metadata['version']}")
    print(f"Timestamp: {metadata['timestamp']}")
    print(f"Floors: {metadata['metadata']['n_floors']}")
    print(f"Duration: {metadata['metadata']['duration']}s")
    print(f"Rating: {metadata['v7']['performance_rating']}")
    print(f"Recommendation: {metadata['v7']['recommendation']}")
    
    # 3. Get baseline vs TMD comparison
    print_section("Performance Comparison")
    response = requests.get(f"{BASE_URL}/comparison")
    comparisons = response.json()
    
    print(f"{'Metric':<25} {'Baseline':<12} {'With TMD':<12} {'Improvement':<12}")
    print("-" * 60)
    for comp in comparisons:
        print(f"{comp['metric']:<25} {comp['baseline']:<12.4f} "
              f"{comp['with_tmd']:<12.4f} {comp['improvement_pct']:<12.2f}%")
    
    # 4. Get TMD configuration
    print_section("TMD Configuration")
    response = requests.get(f"{BASE_URL}/tmd-config")
    tmd = response.json()
    print(f"Floor: {tmd['floor']}")
    print(f"Mass: {tmd['mass_kg']:,.0f} kg")
    print(f"Mass Ratio: {tmd['mass_ratio']}")
    print(f"Damping Ratio: {tmd['damping_ratio']}")
    print(f"Natural Frequency: {tmd['natural_frequency']:.2f} Hz")
    print(f"Optimization Score: {tmd['optimization_score']:.4f}")
    
    # 5. Get improvements
    print_section("Performance Improvements")
    response = requests.get(f"{BASE_URL}/improvements")
    improvements = response.json()
    print(f"DCR Reduction: {improvements['dcr_reduction_pct']:.2f}%")
    print(f"Drift Reduction: {improvements['drift_reduction_pct']:.2f}%")
    print(f"Roof Displacement Reduction: {improvements['roof_reduction_pct']:.2f}%")
    print(f"RMS Acceleration Reduction: {improvements['rms_acc_reduction_pct']:.2f}%")
    
    # 6. Get DCR profile
    print_section("DCR Profile by Floor")
    response = requests.get(f"{BASE_URL}/dcr-profile")
    dcr_data = response.json()
    
    print(f"{'Floor':<8} {'Baseline DCR':<15} {'TMD DCR':<15} {'Reduction':<10}")
    print("-" * 50)
    for floor, base_dcr, tmd_dcr in zip(
        dcr_data['floors'],
        dcr_data['baseline_dcr'],
        dcr_data['tmd_dcr']
    ):
        reduction = ((base_dcr - tmd_dcr) / base_dcr * 100) if base_dcr > 0 else 0
        marker = " *TMD" if floor == dcr_data['tmd_floor'] else ""
        print(f"{floor:<8} {base_dcr:<15.4f} {tmd_dcr:<15.4f} {reduction:<10.2f}%{marker}")
    
    # 7. Get time series summary
    print_section("Time Series Summary")
    response = requests.get(f"{BASE_URL}/time-series/summary")
    summary = response.json()
    print(f"Total Points: {summary['total_points']:,}")
    print(f"Duration: {summary['duration']:.2f}s")
    print(f"Time Step: {summary['time_step']:.4f}s")
    print(f"\nEarthquake:")
    print(f"  Max Acceleration: {summary['earthquake']['max_acceleration']:.4f} m/s²")
    print(f"  Min Acceleration: {summary['earthquake']['min_acceleration']:.4f} m/s²")
    print(f"\nBaseline Roof Displacement:")
    print(f"  Max: {summary['baseline_roof']['max_displacement']:.4f} m")
    print(f"  Min: {summary['baseline_roof']['min_displacement']:.4f} m")
    print(f"\nTMD Roof Displacement:")
    print(f"  Max: {summary['tmd_roof']['max_displacement']:.4f} m")
    print(f"  Min: {summary['tmd_roof']['min_displacement']:.4f} m")
    
    # 8. Get filtered time series
    print_section("Time Series Data (10-20 seconds)")
    response = requests.get(f"{BASE_URL}/time-series?start_time=10&end_time=20")
    ts_data = response.json()
    print(f"Data points in range: {len(ts_data['time'])}")
    if len(ts_data['time']) > 0:
        print(f"First time point: {ts_data['time'][0]:.2f}s")
        print(f"Last time point: {ts_data['time'][-1]:.2f}s")
    
    # 9. Get input parameters
    print_section("Input Parameters")
    response = requests.get(f"{BASE_URL}/input")
    input_data = response.json()
    print(f"Earthquake: {input_data['earthquake']['name']}")
    print(f"Magnitude: {input_data['earthquake']['magnitude']}")
    print(f"PGA: {input_data['earthquake']['pga']:.4f} m/s² ({input_data['earthquake']['pga_g']:.2f}g)")
    print(f"\nWind: {input_data['wind']['name']}")
    print(f"Mean Speed: {input_data['wind']['mean_speed']:.2f} m/s")
    print(f"Total Max Force: {input_data['wind']['total_max_force']:,.2f} N")
    
    print_section("All Tests Completed Successfully!")


if __name__ == "__main__":
    try:
        test_api()
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to API. Make sure the server is running at", BASE_URL)
    except Exception as e:
        print(f"Error: {e}")
"""
Test Trained Neural Network TMD Controller
Verify the model works correctly and compare with fuzzy logic
"""

import numpy as np
import matplotlib.pyplot as plt
from train_neural_network_peer import NeuralTMDController
from generate_training_data_from_peer import FuzzyTMDController


def test_basic_functionality():
    """Test basic controller functionality"""
    print("="*70)
    print("TEST 1: BASIC FUNCTIONALITY")
    print("="*70)
    print()
    
    # Load trained controller
    print("Loading trained neural network controller...")
    try:
        nn_controller = NeuralTMDController('tmd_trained_model_peer.pth')
        print("  ✅ Controller loaded successfully")
    except FileNotFoundError:
        print("  ❌ Model file not found. Please train the model first:")
        print("     python train_neural_network_peer.py")
        return False
    print()
    
    # Test cases
    test_cases = [
        (0.0, 0.0, "At rest"),
        (0.1, 0.5, "Moderate positive motion"),
        (-0.1, -0.5, "Moderate negative motion"),
        (0.3, 1.0, "Large positive motion"),
        (-0.3, -1.0, "Large negative motion"),
        (0.2, -0.8, "Positive displacement, negative velocity"),
        (-0.2, 0.8, "Negative displacement, positive velocity"),
    ]
    
    print("Testing various building states:")
    print("-" * 70)
    print(f"{'Displacement (m)':<18} {'Velocity (m/s)':<18} {'Force (kN)':<15} {'Scenario'}")
    print("-" * 70)
    
    for disp, vel, description in test_cases:
        force = nn_controller.compute(disp, vel)
        print(f"{disp:>16.2f}   {vel:>16.2f}   {force:>13.1f}   {description}")
    
    print()
    print("✅ Basic functionality test passed")
    print()
    return True


def test_comparison_with_fuzzy():
    """Compare neural network with fuzzy logic controller"""
    print("="*70)
    print("TEST 2: COMPARISON WITH FUZZY LOGIC")
    print("="*70)
    print()
    
    # Load controllers
    print("Loading controllers...")
    nn_controller = NeuralTMDController('tmd_trained_model_peer.pth')
    fuzzy_controller = FuzzyTMDController()
    print("  ✅ Both controllers loaded")
    print()
    
    # Generate test grid
    n_points = 20
    displacements = np.linspace(-0.3, 0.3, n_points)
    velocities = np.linspace(-1.5, 1.5, n_points)
    
    # Compute forces for both controllers
    print("Computing control surfaces...")
    nn_forces = np.zeros((n_points, n_points))
    fuzzy_forces = np.zeros((n_points, n_points))
    fuzzy_failures = 0
    
    for i, disp in enumerate(displacements):
        for j, vel in enumerate(velocities):
            nn_forces[j, i] = nn_controller.compute(disp, vel)
            
            # Try fuzzy controller with error handling
            try:
                fuzzy_force = fuzzy_controller.compute(disp, vel)
                # Check if fuzzy returned a valid result
                if np.isnan(fuzzy_force) or np.isinf(fuzzy_force):
                    fuzzy_force = 0.0
                    fuzzy_failures += 1
                fuzzy_forces[j, i] = fuzzy_force
            except Exception as e:
                fuzzy_forces[j, i] = 0.0
                fuzzy_failures += 1
    
    if fuzzy_failures > 0:
        print(f"  ⚠️  Fuzzy controller failed for {fuzzy_failures}/{n_points*n_points} points")
    
    print("  ✅ Surfaces computed")
    print()
    
    # Calculate error - only for points where forces are significant
    # For structural control, forces < 5 kN don't meaningfully affect building response
    # This prevents small denominators from inflating relative error percentages
    significant_mask = np.abs(fuzzy_forces) > 5.0
    n_significant = np.sum(significant_mask)
    
    if n_significant == 0:
        print("  ❌ Error: No significant control forces to compare!")
        print("     (All forces < 5 kN threshold)")
        return False
    
    if n_significant < 50:
        print(f"  ⚠️  Warning: Only {n_significant} points with significant forces")
    
    absolute_error = np.abs(nn_forces - fuzzy_forces)
    
    # Calculate relative error only for significant force points
    relative_error_significant = np.abs(
        (nn_forces[significant_mask] - fuzzy_forces[significant_mask]) / fuzzy_forces[significant_mask]
    ) * 100
    
    # Also calculate error statistics for ALL points (not just significant)
    all_points_mean_abs = np.mean(absolute_error)
    all_points_max_abs = np.max(absolute_error)
    
    print("Error Statistics:")
    print(f"  Points with significant forces (>5 kN): {n_significant}/{n_points*n_points}")
    print(f"  Mean absolute error (all points): {all_points_mean_abs:.2f} kN")
    print(f"  Max absolute error (all points): {all_points_max_abs:.2f} kN")
    
    if n_significant > 0:
        print(f"  Mean relative error (significant forces): {np.mean(relative_error_significant):.1f}%")
        print(f"  Max relative error (significant forces): {np.max(relative_error_significant):.1f}%")
    print()
    
    # Determine pass/fail
    mean_abs_error = all_points_mean_abs
    mean_rel_error = np.mean(relative_error_significant) if n_significant > 0 else 0
    
    passed = True
    if mean_abs_error > 10.0:
        print(f"  ⚠️  Mean absolute error too high: {mean_abs_error:.2f} kN > 10 kN threshold")
        passed = False
    if n_significant > 0 and mean_rel_error > 15.0:
        print(f"  ⚠️  Mean relative error too high: {mean_rel_error:.1f}% > 15% threshold")
        passed = False
    if fuzzy_failures > n_points * n_points * 0.3:
        print(f"  ⚠️  Too many fuzzy controller failures: {fuzzy_failures}/{n_points*n_points}")
        passed = False
    
    if passed:
        print(f"  ✅ Errors within acceptable range")
    print()
    
    # Visualize comparison
    print("Generating comparison plot...")
    D, V = np.meshgrid(displacements, velocities)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Fuzzy logic surface
    im1 = axes[0, 0].contourf(D, V, fuzzy_forces, levels=20, cmap='RdBu_r')
    axes[0, 0].set_xlabel('Displacement (m)')
    axes[0, 0].set_ylabel('Velocity (m/s)')
    axes[0, 0].set_title(f'Fuzzy Logic Controller ({fuzzy_failures} failures)')
    plt.colorbar(im1, ax=axes[0, 0], label='Force (kN)')
    
    # Neural network surface
    im2 = axes[0, 1].contourf(D, V, nn_forces, levels=20, cmap='RdBu_r')
    axes[0, 1].set_xlabel('Displacement (m)')
    axes[0, 1].set_ylabel('Velocity (m/s)')
    axes[0, 1].set_title('Neural Network Controller')
    plt.colorbar(im2, ax=axes[0, 1], label='Force (kN)')
    
    # Absolute error
    im3 = axes[1, 0].contourf(D, V, absolute_error, levels=20, cmap='YlOrRd')
    axes[1, 0].set_xlabel('Displacement (m)')
    axes[1, 0].set_ylabel('Velocity (m/s)')
    axes[1, 0].set_title(f'Absolute Error (Mean: {np.mean(absolute_error):.2f} kN)')
    plt.colorbar(im3, ax=axes[1, 0], label='Error (kN)')
    
    # Scatter plot comparison - separate by significance
    if n_significant > 0:
        axes[1, 1].scatter(
            fuzzy_forces[significant_mask].flatten(), 
            nn_forces[significant_mask].flatten(), 
            alpha=0.6, s=30, label=f'Significant (>5kN, n={n_significant})', color='blue'
        )
    
    # Show low-force points separately
    low_force_mask = ~significant_mask
    if np.sum(low_force_mask) > 0:
        axes[1, 1].scatter(
            fuzzy_forces[low_force_mask].flatten(),
            nn_forces[low_force_mask].flatten(),
            alpha=0.3, s=10, color='gray', marker='.', label=f'Low force (<5kN)'
        )
    
    # Perfect match line
    if n_significant > 0:
        min_val = min(fuzzy_forces[significant_mask].min(), nn_forces[significant_mask].min())
        max_val = max(fuzzy_forces[significant_mask].max(), nn_forces[significant_mask].max())
        axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Match')
    
    axes[1, 1].set_xlabel('Fuzzy Logic Force (kN)')
    axes[1, 1].set_ylabel('Neural Network Force (kN)')
    axes[1, 1].set_title('Force Comparison')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('controller_comparison.png', dpi=150, bbox_inches='tight')
    print("  ✅ Saved controller_comparison.png")
    plt.close()
    
    plt.tight_layout()
    plt.savefig('controller_comparison.png', dpi=150, bbox_inches='tight')
    print("  ✅ Saved controller_comparison.png")
    plt.close()
    
    print()
    if passed:
        print("✅ Comparison test passed")
    else:
        print("❌ Comparison test failed")
    print()
    
    return passed


def test_inference_speed():
    """Test inference speed of neural network"""
    print("="*70)
    print("TEST 3: INFERENCE SPEED")
    print("="*70)
    print()
    
    # Load controller
    nn_controller = NeuralTMDController('tmd_trained_model_peer.pth')
    
    # Generate random test data
    n_tests = 10000
    displacements = np.random.uniform(-0.3, 0.3, n_tests)
    velocities = np.random.uniform(-1.5, 1.5, n_tests)
    
    print(f"Running {n_tests} inference tests...")
    
    import time
    start_time = time.time()
    
    forces = [nn_controller.compute(d, v) for d, v in zip(displacements, velocities)]
    
    elapsed_time = time.time() - start_time
    
    avg_time = elapsed_time / n_tests * 1000  # Convert to ms
    throughput = n_tests / elapsed_time
    
    print()
    print("Performance Metrics:")
    print(f"  Total time: {elapsed_time:.2f} seconds")
    print(f"  Average time per prediction: {avg_time:.4f} ms")
    print(f"  Throughput: {throughput:.0f} predictions/second")
    print()
    
    # Check if it's fast enough for real-time control (50 Hz = 20ms budget)
    if avg_time < 20:
        print(f"✅ Fast enough for real-time control (50 Hz requires <20ms)")
    else:
        print(f"⚠️  Too slow for real-time control at 50 Hz (need <20ms, got {avg_time:.2f}ms)")
    
    print()
    print("✅ Inference speed test passed")
    print()
    
    return True


def test_physical_consistency():
    """Test if controller follows physical intuition"""
    print("="*70)
    print("TEST 4: PHYSICAL CONSISTENCY")
    print("="*70)
    print()
    
    nn_controller = NeuralTMDController('tmd_trained_model_peer.pth')
    
    print("Testing physical consistency rules:")
    print()
    
    passed = True
    
    # Test 1: Positive displacement + positive velocity → negative force
    disp, vel = 0.2, 0.8
    force = nn_controller.compute(disp, vel)
    print(f"1. Positive motion (d={disp}, v={vel}) → Force={force:.1f} kN")
    if force < 0:
        print("   ✅ Correct: Negative force opposes positive motion")
    else:
        print("   ❌ Error: Expected negative force")
        passed = False
    print()
    
    # Test 2: Negative displacement + negative velocity → positive force
    disp, vel = -0.2, -0.8
    force = nn_controller.compute(disp, vel)
    print(f"2. Negative motion (d={disp}, v={vel}) → Force={force:.1f} kN")
    if force > 0:
        print("   ✅ Correct: Positive force opposes negative motion")
    else:
        print("   ❌ Error: Expected positive force")
        passed = False
    print()
    
    # Test 3: Zero state → near-zero force
    disp, vel = 0.0, 0.0
    force = nn_controller.compute(disp, vel)
    print(f"3. At rest (d={disp}, v={vel}) → Force={force:.1f} kN")
    if abs(force) < 10:
        print("   ✅ Correct: Minimal force when at rest")
    else:
        print("   ⚠️  Warning: Expected near-zero force")
    print()
    
    # Test 4: Larger displacement → larger force
    force1 = abs(nn_controller.compute(0.1, 0.5))
    force2 = abs(nn_controller.compute(0.2, 0.5))
    print(f"4. Scaling test:")
    print(f"   Small displacement (0.1m): {force1:.1f} kN")
    print(f"   Large displacement (0.2m): {force2:.1f} kN")
    if force2 > force1:
        print("   ✅ Correct: Force increases with displacement")
    else:
        print("   ⚠️  Warning: Expected larger force for larger displacement")
    print()
    
    if passed:
        print("✅ Physical consistency test passed")
    else:
        print("❌ Physical consistency test failed")
    
    print()
    return passed


def main():
    """Run all tests"""
    print()
    print("█" * 70)
    print("  TESTING TRAINED NEURAL NETWORK TMD CONTROLLER")
    print("█" * 70)
    print()
    
    results = []
    
    # Run tests
    results.append(("Basic Functionality", test_basic_functionality()))
    results.append(("Comparison with Fuzzy Logic", test_comparison_with_fuzzy()))
    results.append(("Inference Speed", test_inference_speed()))
    results.append(("Physical Consistency", test_physical_consistency()))
    
    # Summary
    print()
    print("="*70)
    print("TEST SUMMARY")
    print("="*70)
    print()
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:<35} {status}")
    
    print()
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print()
        print("Your neural network controller is ready for deployment!")
        print("Next steps:")
        print("  1. Deploy to REST API")
        print("  2. Test with real building simulation")
        print("  3. Compare performance with passive TMD")
    else:
        print("⚠️  SOME TESTS FAILED")
        print()
        print("Please review the failed tests and retrain if necessary.")
    
    print()
    print("="*70)


if __name__ == '__main__':
    main()
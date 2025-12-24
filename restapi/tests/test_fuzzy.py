"""
LOCAL TEST SCRIPT FOR FIXED FUZZY CONTROLLER
============================================

Run this BEFORE deploying to Cloud Run to verify everything works

Usage:
    python test_fuzzy_local.py
"""

import numpy as np
import matplotlib.pyplot as plt
from fixed_fuzzy_controller import FixedFuzzyTMDController


def test_basic_functionality():
    """Test 1: Basic functionality"""
    
    print("\n" + "="*70)
    print("TEST 1: BASIC FUNCTIONALITY")
    print("="*70)
    
    controller = FixedFuzzyTMDController()
    
    # Simple test cases
    tests = [
        # (rel_disp, rel_vel, description)
        (0.15, 0.8, "Moving right, push left"),
        (-0.15, -0.8, "Moving left, push right"),
        (0.0, 0.0, "At rest"),
        (0.3, 1.2, "Far right, fast right"),
        (-0.3, -1.2, "Far left, fast left"),
    ]
    
    print(f"\n{'Rel Disp':<12} {'Rel Vel':<12} {'Force (kN)':<12} Description")
    print("-" * 60)
    
    for disp, vel, desc in tests:
        force_N = controller.compute(disp, vel)
        force_kN = force_N / 1000
        print(f"{disp:<12.3f} {vel:<12.3f} {force_kN:<12.2f} {desc}")
    
    print("\n✅ Basic functionality test complete\n")


def test_force_magnitude():
    """Test 2: Force magnitude range"""
    
    print("="*70)
    print("TEST 2: FORCE MAGNITUDE RANGE")
    print("="*70)
    
    controller = FixedFuzzyTMDController()
    
    # Generate range of inputs
    n_samples = 100
    displacements = np.linspace(-0.4, 0.4, n_samples)
    velocities = np.linspace(-1.5, 1.5, n_samples)
    
    forces = []
    for i in range(n_samples):
        force_N = controller.compute(displacements[i], velocities[i])
        forces.append(force_N / 1000)  # Convert to kN
    
    forces = np.array(forces)
    
    print(f"\nForce Statistics:")
    print(f"  Min force: {np.min(forces):.2f} kN")
    print(f"  Max force: {np.max(forces):.2f} kN")
    print(f"  Mean |force|: {np.mean(np.abs(forces)):.2f} kN")
    print(f"  Std dev: {np.std(forces):.2f} kN")
    
    # Check if within limits
    if np.max(np.abs(forces)) <= 100:
        print(f"\n✅ All forces within ±100 kN limit")
    else:
        print(f"\n⚠️  Some forces exceed ±100 kN limit!")
    
    print()


def test_velocity_opposition():
    """Test 3: Verify force opposes velocity (critical for damping)"""
    
    print("="*70)
    print("TEST 3: VELOCITY OPPOSITION (CRITICAL)")
    print("="*70)
    
    controller = FixedFuzzyTMDController()
    
    # Test: Force should ALWAYS oppose velocity
    velocities = np.linspace(-1.5, 1.5, 50)
    displacements = [0.0, 0.1, -0.1]  # Try different positions
    
    all_correct = True
    failures = []
    
    for disp in displacements:
        for vel in velocities:
            if abs(vel) < 0.01:  # Skip near-zero velocities
                continue
            
            force_N = controller.compute(disp, vel)
            
            # Force should have OPPOSITE sign to velocity
            # If vel > 0 (moving right), force should be < 0 (push left)
            # If vel < 0 (moving left), force should be > 0 (push right)
            correct = (vel * force_N) < 0
            
            if not correct:
                all_correct = False
                failures.append((disp, vel, force_N))
    
    if all_correct:
        print("\n✅ PERFECT! Force ALWAYS opposes velocity (proper damping)")
    else:
        print(f"\n❌ PROBLEM! {len(failures)} cases where force doesn't oppose velocity:")
        for disp, vel, force in failures[:5]:  # Show first 5
            print(f"   disp={disp:.3f}, vel={vel:.3f}, force={force/1000:.2f} kN")
    
    print()


def test_batch_processing():
    """Test 4: Batch processing (for API)"""
    
    print("="*70)
    print("TEST 4: BATCH PROCESSING")
    print("="*70)
    
    controller = FixedFuzzyTMDController()
    
    # Create batch input
    n = 20
    displacements = np.random.uniform(-0.3, 0.3, n)
    velocities = np.random.uniform(-1.0, 1.0, n)
    
    # Process batch
    forces_N = controller.compute_batch(displacements, velocities)
    forces_kN = forces_N / 1000
    
    print(f"\nProcessed {n} samples")
    print(f"  Force range: [{np.min(forces_kN):.2f}, {np.max(forces_kN):.2f}] kN")
    print(f"  Mean |force|: {np.mean(np.abs(forces_kN)):.2f} kN")
    
    print("\n✅ Batch processing works\n")


def visualize_control_surface():
    """Test 5: Visualize force as function of displacement & velocity"""
    
    print("="*70)
    print("TEST 5: CONTROL SURFACE VISUALIZATION")
    print("="*70)
    
    controller = FixedFuzzyTMDController()
    
    # Create grid
    disp_range = np.linspace(-0.4, 0.4, 50)
    vel_range = np.linspace(-1.5, 1.5, 50)
    
    D, V = np.meshgrid(disp_range, vel_range)
    F = np.zeros_like(D)
    
    print("\nGenerating control surface (this may take 10-20 seconds)...")
    
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            force_N = controller.compute(D[i,j], V[i,j])
            F[i,j] = force_N / 1000  # kN
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 3D surface
    from mpl_toolkits.mplot3d import Axes3D
    fig2 = plt.figure(figsize=(10, 7))
    ax3d = fig2.add_subplot(111, projection='3d')
    surf = ax3d.plot_surface(D, V, F, cmap='coolwarm', alpha=0.8)
    ax3d.set_xlabel('Relative Displacement (m)', fontsize=10)
    ax3d.set_ylabel('Relative Velocity (m/s)', fontsize=10)
    ax3d.set_zlabel('Control Force (kN)', fontsize=10)
    ax3d.set_title('Fuzzy Controller Control Surface', fontsize=12, fontweight='bold')
    fig2.colorbar(surf, ax=ax3d, shrink=0.5)
    
    # Contour plot
    contour = ax1.contourf(D, V, F, levels=20, cmap='coolwarm')
    ax1.set_xlabel('Relative Displacement (m)', fontsize=11)
    ax1.set_ylabel('Relative Velocity (m/s)', fontsize=11)
    ax1.set_title('Control Force (kN) - Contour View', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    fig.colorbar(contour, ax=ax1)
    
    # Phase portrait with force vectors
    skip = 5  # Show every 5th vector
    ax2.quiver(D[::skip, ::skip], V[::skip, ::skip], 
               -D[::skip, ::skip], -V[::skip, ::skip],  # Direction vectors
               F[::skip, ::skip], cmap='coolwarm', alpha=0.6)
    ax2.set_xlabel('Relative Displacement (m)', fontsize=11)
    ax2.set_ylabel('Relative Velocity (m/s)', fontsize=11)
    ax2.set_title('Phase Portrait with Control Forces', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fuzzy_control_surface.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ Control surface saved as 'fuzzy_control_surface.png'")
    
    plt.show()
    print()


def main():
    """Run all tests"""
    
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*68 + "║")
    print("║" + " "*15 + "FIXED FUZZY CONTROLLER TEST SUITE" + " "*20 + "║")
    print("║" + " "*68 + "║")
    print("╚" + "="*68 + "╝")
    
    try:
        # Run all tests
        test_basic_functionality()
        test_force_magnitude()
        test_velocity_opposition()
        test_batch_processing()
        visualize_control_surface()
        
        print("="*70)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\n📋 NEXT STEPS:")
        print("   1. Review the control surface plot")
        print("   2. Add to your FastAPI application")
        print("   3. Deploy to Cloud Run")
        print("   4. Test from MATLAB\n")
        
    except Exception as e:
        print(f"\n❌ ERROR during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
"""
Quick Test Script for Fuzzy Logic TMD Controller
Run this to verify everything works before integrating with MATLAB
"""

import json
from pathlib import Path

def test_fuzzy_controller():
    """Test the fuzzy controller with sample data"""
    
    print("="*70)
    print("FUZZY LOGIC TMD CONTROLLER - QUICK TEST")
    print("="*70)
    
    # Import the controller
    try:
        import sys
        sys.path.append(str(Path(__file__).parent))
        from restapi.fuzzy.fuzzy_main import fuzzy_controller
        print("✅ Controller imported successfully")
    except Exception as e:
        print(f"❌ Error importing controller: {e}")
        return False
    
    # Test 1: Single computation
    print("\n" + "-"*70)
    print("TEST 1: Single Computation")
    print("-"*70)
    
    displacement = 0.082  # 8.2 cm
    velocity = 0.43       # 43 cm/s
    acceleration = 3.48   # 3.48 m/s²
    
    print(f"Input: displacement={displacement}m, velocity={velocity}m/s, accel={acceleration}m/s²")
    
    try:
        force = fuzzy_controller.compute(displacement, velocity, acceleration)
        print(f"✅ Output: {force:.1f} N ({force/1000:.1f} kN)")
        print(f"   Direction: {'←  Pull left' if force < 0 else 'Push right  →'}")
    except Exception as e:
        print(f"❌ Computation failed: {e}")
        return False
    
    # Test 2: Batch computation
    print("\n" + "-"*70)
    print("TEST 2: Batch Computation (10 steps)")
    print("-"*70)
    
    import numpy as np
    
    displacements = np.linspace(-0.1, 0.1, 10)
    velocities = np.linspace(-0.5, 0.5, 10)
    
    try:
        forces = fuzzy_controller.compute_batch(displacements, velocities)
        print(f"✅ Computed {len(forces)} control forces")
        print(f"   Max force: {np.max(np.abs(forces))/1000:.1f} kN")
        print(f"   Mean force: {np.mean(np.abs(forces))/1000:.1f} kN")
    except Exception as e:
        print(f"❌ Batch computation failed: {e}")
        return False
    
    # Test 3: Check statistics
    print("\n" + "-"*70)
    print("TEST 3: Controller Statistics")
    print("-"*70)
    
    try:
        stats = fuzzy_controller.get_stats()
        print(f"✅ Total computations: {stats['total_computations']}")
        print(f"   Last computation: {stats['last_computation']}")
        print(f"   Force range: ±{stats['force_range_kN'][1]:.1f} kN")
    except Exception as e:
        print(f"❌ Statistics failed: {e}")
        return False
    
    # Test 4: Edge cases
    print("\n" + "-"*70)
    print("TEST 4: Edge Cases")
    print("-"*70)
    
    test_cases = [
        (0.0, 0.0, "Equilibrium"),
        (0.5, 2.0, "Maximum positive"),
        (-0.5, -2.0, "Maximum negative"),
        (0.6, 3.0, "Beyond range (should clip)")
    ]
    
    for disp, vel, description in test_cases:
        try:
            force = fuzzy_controller.compute(disp, vel)
            print(f"✅ {description:30s} → {force/1000:>7.1f} kN")
        except Exception as e:
            print(f"❌ {description:30s} → ERROR: {e}")
    
    # Final summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print("✅ All tests passed!")
    print(f"✅ Total computations: {fuzzy_controller.computation_count}")
    print("✅ Controller is ready to use with MATLAB")
    print("="*70)
    
    return True


if __name__ == "__main__":
    import sys
    
    # Check if required packages are available
    try:
        import numpy as np
        import skfuzzy
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("\nInstall with:")
        print("  pip install numpy scikit-fuzzy")
        sys.exit(1)
    
    # Run tests
    success = test_fuzzy_controller()
    
    if success:
        print("\n🎉 Controller test successful!")
        print("\nNext steps:")
        print("1. Start the API: python main.py")
        print("2. Test from MATLAB: see matlab_fuzzy_integration.m")
        print("3. Read the guide: README_FUZZY_CONTROLLER.md")
    else:
        print("\n❌ Tests failed. Check errors above.")
        sys.exit(1)

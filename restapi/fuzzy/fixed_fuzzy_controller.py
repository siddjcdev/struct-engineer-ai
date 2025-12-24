"""
FIXED FUZZY LOGIC TMD CONTROLLER
================================

Fixes from original:
1. Uses RELATIVE motion (TMD - roof), not absolute
2. Proper fuzzy rules that oppose motion
3. Returns forces in Newtons
4. Ready for Cloud Run API deployment

Author: Siddharth
Date: December 2025
"""

from typing import List, Optional
import numpy as np
from pydantic import BaseModel, Field
import skfuzzy as fuzz
from skfuzzy import control as ctrl


class FixedFuzzyTMDController:
    """
    Fuzzy Logic Controller for TMD with correct physics
    """
    
    def __init__(self):
        print("Initializing Fixed Fuzzy TMD Controller...")
        
        # ================================================================
        # DEFINE INPUT/OUTPUT UNIVERSES
        # ================================================================
        
        # Input 1: Relative displacement (TMD - roof) in meters
        # Negative = TMD is left of roof, Positive = TMD is right of roof
        self.displacement = ctrl.Antecedent(np.linspace(-0.5, 0.5, 100), 'displacement')
        
        # Input 2: Relative velocity (TMD - roof) in m/s
        # Negative = TMD moving left relative to roof, Positive = moving right
        self.velocity = ctrl.Antecedent(np.linspace(-2.0, 2.0, 100), 'velocity')
        
        # Output: Control force in Newtons
        # Negative = push left, Positive = push right
        self.force = ctrl.Consequent(np.linspace(-100000, 100000, 200), 'force')
        
        # ================================================================
        # DEFINE MEMBERSHIP FUNCTIONS
        # ================================================================
        
        # Displacement fuzzy sets
        self.displacement['large_left'] = fuzz.trimf(self.displacement.universe, [-0.5, -0.5, -0.2])
        self.displacement['medium_left'] = fuzz.trimf(self.displacement.universe, [-0.3, -0.15, -0.05])
        self.displacement['small_left'] = fuzz.trimf(self.displacement.universe, [-0.1, -0.03, 0])
        self.displacement['zero'] = fuzz.trimf(self.displacement.universe, [-0.05, 0, 0.05])
        self.displacement['small_right'] = fuzz.trimf(self.displacement.universe, [0, 0.03, 0.1])
        self.displacement['medium_right'] = fuzz.trimf(self.displacement.universe, [0.05, 0.15, 0.3])
        self.displacement['large_right'] = fuzz.trimf(self.displacement.universe, [0.2, 0.5, 0.5])
        
        # Velocity fuzzy sets (PRIMARY damping control)
        self.velocity['fast_left'] = fuzz.trimf(self.velocity.universe, [-2.0, -2.0, -0.8])
        self.velocity['medium_left'] = fuzz.trimf(self.velocity.universe, [-1.2, -0.5, -0.2])
        self.velocity['slow_left'] = fuzz.trimf(self.velocity.universe, [-0.4, -0.1, 0])
        self.velocity['zero'] = fuzz.trimf(self.velocity.universe, [-0.2, 0, 0.2])
        self.velocity['slow_right'] = fuzz.trimf(self.velocity.universe, [0, 0.1, 0.4])
        self.velocity['medium_right'] = fuzz.trimf(self.velocity.universe, [0.2, 0.5, 1.2])
        self.velocity['fast_right'] = fuzz.trimf(self.velocity.universe, [0.8, 2.0, 2.0])
        
        # Force fuzzy sets
        self.force['large_left'] = fuzz.trimf(self.force.universe, [-100000, -100000, -60000])
        self.force['medium_left'] = fuzz.trimf(self.force.universe, [-80000, -50000, -20000])
        self.force['small_left'] = fuzz.trimf(self.force.universe, [-40000, -15000, -5000])
        self.force['zero'] = fuzz.trimf(self.force.universe, [-10000, 0, 10000])
        self.force['small_right'] = fuzz.trimf(self.force.universe, [5000, 15000, 40000])
        self.force['medium_right'] = fuzz.trimf(self.force.universe, [20000, 50000, 80000])
        self.force['large_right'] = fuzz.trimf(self.force.universe, [60000, 100000, 100000])
        
        # ================================================================
        # DEFINE FUZZY RULES (CORRECT PHYSICS!)
        # ================================================================
        
        rules = []
        
        # CRITICAL PRINCIPLE: Force opposes VELOCITY (primary damping)
        # If moving right → push left (negative force)
        # If moving left → push right (positive force)
        
        # -------------------- VELOCITY-BASED RULES (Primary) --------------------
        
        # Moving FAST RIGHT → Push HARD LEFT
        rules.append(ctrl.Rule(self.velocity['fast_right'], self.force['large_left']))
        
        # Moving MEDIUM RIGHT → Push MEDIUM LEFT
        rules.append(ctrl.Rule(self.velocity['medium_right'], self.force['medium_left']))
        
        # Moving SLOW RIGHT → Push SMALL LEFT
        rules.append(ctrl.Rule(self.velocity['slow_right'], self.force['small_left']))
        
        # Nearly stationary → Small force
        rules.append(ctrl.Rule(self.velocity['zero'], self.force['zero']))
        
        # Moving SLOW LEFT → Push SMALL RIGHT
        rules.append(ctrl.Rule(self.velocity['slow_left'], self.force['small_right']))
        
        # Moving MEDIUM LEFT → Push MEDIUM RIGHT
        rules.append(ctrl.Rule(self.velocity['medium_left'], self.force['medium_right']))
        
        # Moving FAST LEFT → Push HARD RIGHT
        rules.append(ctrl.Rule(self.velocity['fast_left'], self.force['large_right']))
        
        # -------------------- COMBINED RULES (Velocity + Position) --------------------
        
        # Far RIGHT and moving RIGHT → MAXIMUM push LEFT
        rules.append(ctrl.Rule(
            self.displacement['large_right'] & self.velocity['fast_right'],
            self.force['large_left']
        ))
        
        # Far RIGHT and moving SLOW RIGHT → LARGE push LEFT
        rules.append(ctrl.Rule(
            self.displacement['large_right'] & self.velocity['medium_right'],
            self.force['large_left']
        ))
        
        # Far LEFT and moving LEFT → MAXIMUM push RIGHT
        rules.append(ctrl.Rule(
            self.displacement['large_left'] & self.velocity['fast_left'],
            self.force['large_right']
        ))
        
        # Far LEFT and moving SLOW LEFT → LARGE push RIGHT
        rules.append(ctrl.Rule(
            self.displacement['large_left'] & self.velocity['medium_left'],
            self.force['large_right']
        ))
        
        # Medium RIGHT + Medium velocity RIGHT → Medium force LEFT
        rules.append(ctrl.Rule(
            self.displacement['medium_right'] & self.velocity['medium_right'],
            self.force['medium_left']
        ))
        
        # Medium LEFT + Medium velocity LEFT → Medium force RIGHT
        rules.append(ctrl.Rule(
            self.displacement['medium_left'] & self.velocity['medium_left'],
            self.force['medium_right']
        ))
        
        # -------------------- RESTORING FORCE RULES (Position-based) --------------------
        
        # Far RIGHT but moving LEFT → Gentle push LEFT (let it come back)
        rules.append(ctrl.Rule(
            self.displacement['large_right'] & self.velocity['medium_left'],
            self.force['small_left']
        ))
        
        # Far LEFT but moving RIGHT → Gentle push RIGHT (let it come back)
        rules.append(ctrl.Rule(
            self.displacement['large_left'] & self.velocity['medium_right'],
            self.force['small_right']
        ))
        
        # ================================================================
        # CREATE CONTROL SYSTEM
        # ================================================================
        
        self.control_system = ctrl.ControlSystem(rules)
        self.controller = ctrl.ControlSystemSimulation(self.control_system)
        
        print(f"✅ FUZZY: Fuzzy controller initialized with {len(rules)} rules")
        print(f"   FUZZY: Input ranges: displacement [-0.5, 0.5] m, velocity [-2, 2] m/s")
        print(f"   FUZZY: Output range: force [-100, 100] kN")
    
    
    def compute(self, roof_disp, roof_vel, tmd_disp, tmd_vel):
        """
        Compute control force for given absolute states
        
        Args:
            roof_disp: Roof displacement (meters)
            roof_vel: Roof velocity (m/s)
            tmd_disp: TMD displacement (meters)
            tmd_vel: TMD velocity (m/s)
            
        Returns:
            control_force: Force in Newtons (will be applied with Newton's 3rd law)
        """
        #print(f"FUZZY: Computing force for disp={relative_displacement}, vel={relative_velocity}...")
        # Calculate relative states (TMD relative to roof)
        relative_displacement = tmd_disp - roof_disp
        relative_velocity = tmd_vel - roof_vel
        
        #print(f"FUZZY: roof_disp={roof_disp:.6f}, roof_vel={roof_vel:.6f}")
        #print(f"FUZZY: tmd_disp={tmd_disp:.6f}, tmd_vel={tmd_vel:.6f}")
        #print(f"FUZZY: relative_disp={relative_displacement:.6f}, relative_vel={relative_velocity:.6f}")
        
        # Clamp inputs to valid range
        disp = np.clip(relative_displacement, -0.5, 0.5)
        vel = np.clip(relative_velocity, -2.0, 2.0)
        
        # Handle edge case: if both near zero, return zero force
        if abs(disp) < 1e-6 and abs(vel) < 1e-6:
            print(f"FUZZY: Near-zero state, returning 0 N")
            return 0.0
        
        try:
            # Set inputs
            self.controller.input['displacement'] = disp
            self.controller.input['velocity'] = vel
            
            #print(f"FUZZY: Inputs set - disp: {disp:.6f}, vel: {vel:.6f}")
            
            # Compute
            self.controller.compute()
            
            #print(f"FUZZY: Computation complete - output force: {self.controller.output['force']:.2f} N")
            
            # Get output force in Newtons
            force_N = float(self.controller.output['force'])
            #print(f"FUZZY: Raw output force: {force_N:.2f} N")
            
            # Clamp to physical limits (±100 kN)
            force_N = np.clip(force_N, -100000, 100000)
            #print(f"FUZZY: Clamped output force: {force_N:.2f} N")
            
            return force_N
        
        except Exception as e:
            print(f"Warning: Fuzzy computation failed for disp={disp:.6f}, vel={vel:.6f}: {e}")
            # Fallback to simple PD control if fuzzy fails
            Kp = 50000  # N/m
            Kd = 20000  # N·s/m
            fallback_force = -Kp * disp - Kd * vel
            print(f"FUZZY: Using fallback PD control: {fallback_force:.2f} N")
            return fallback_force
    
    # def compute_old(self, relative_displacement, relative_velocity):
    #     """
    #     Compute control force for given relative state
        
    #     Args:
    #         relative_displacement: TMD displacement - roof displacement (meters)
    #         relative_velocity: TMD velocity - roof velocity (m/s)
            
    #     Returns:
    #         control_force: Force in Newtons (will be applied with Newton's 3rd law)
    #     """
    #     print(f"FUZZY: Computing force for disp={relative_displacement}, vel={relative_velocity}...")
    #     # Clamp inputs to valid range
    #     disp = np.clip(relative_displacement, -0.5, 0.5)
    #     vel = np.clip(relative_velocity, -2.0, 2.0)
        
    #     # Handle edge case: if both near zero, return zero force
    #     if abs(disp) < 1e-6 and abs(vel) < 1e-6:
    #         return 0.0
        
    #     try:
    #         # Set inputs
    #         self.controller.input['displacement'] = disp
    #         self.controller.input['velocity'] = vel
            
    #         print(f"FUZZY: Inputs set - disp: {disp}, vel: {vel}")
    #         # Compute
    #         self.controller.compute()
            
    #         print(f"FUZZY: Computation complete - output force: {self.controller.output['force']} N")
    #         # Get output force in Newtons
    #         force_N = float(self.controller.output['force'])
    #         print(f"FUZZY: Raw output force: {force_N} N")
    #         # Clamp to physical limits (±100 kN)
    #         force_N = np.clip(force_N, -100000, 100000)
    #         print(f"FUZZY: Clamped output force: {force_N} N")
    #         return force_N
            
    #     except Exception as e:
    #         print(f"Warning: Fuzzy computation failed for disp={disp}, vel={vel}: {e}")
    #         # Fallback to simple PD control if fuzzy fails
    #         Kp = 50000  # N/m
    #         Kd = 20000  # N·s/m
    #         return -Kp * disp - Kd * vel
    
    def compute_batch(self, roof_displacements, roof_velocities, tmd_displacements, tmd_velocities):
        """
        Compute control forces for batch of states
        
        Args:
            roof_displacements: Array of roof displacements (meters)
            roof_velocities: Array of roof velocities (m/s)
            tmd_displacements: Array of TMD displacements (meters)
            tmd_velocities: Array of TMD velocities (m/s)
            
        Returns:
            forces: Array of forces in Newtons
        """
        n = len(roof_displacements)
        forces = np.zeros(n)
        
        for i in range(n):
            forces[i] = self.compute(
                roof_displacements[i],
                roof_velocities[i],
                tmd_displacements[i],
                tmd_velocities[i]
            )
        
        return forces

    # def compute_batch_OLD(self, relative_displacements, relative_velocities):
    #     """
    #     Compute control forces for batch of states
        
    #     Args:
    #         relative_displacements: Array of (TMD - roof) displacements
    #         relative_velocities: Array of (TMD - roof) velocities
            
    #     Returns:
    #         forces: Array of control forces in Newtons
    #     """
        
    #     displacements = np.array(relative_displacements)
    #     velocities = np.array(relative_velocities)
        
    #     forces = np.zeros(len(displacements))
        
    #     for i in range(len(displacements)):
    #         forces[i] = self.compute(displacements[i], velocities[i])
        
    #     return forces

    def get_stats(self):
        """Get controller statistics"""
        return {
            #"displacement_range_m": self.displacement.universe,
            #"velocity_range_ms": self.velocity.universe,
            #"force_range_kN": self.force.universe,
            "status": "active"
        }
# ================================================================
# REQUEST/RESPONSE MODELS
# ================================================================

class FuzzyBatchRequest(BaseModel):
    """
    Batch prediction request for fuzzy controller
    """
    roof_displacements: List[float] = Field(..., description="Roof displacements (m)")
    roof_velocities: List[float] = Field(..., description="Roof velocities (m/s)")
    tmd_displacements: List[float] = Field(..., description="TMD displacements (m)")
    tmd_velocities: List[float] = Field(..., description="TMD velocities (m/s)")
    


class FuzzyBatchResponse(BaseModel):
    """Batch prediction response"""
    forces: List[float]  # Forces in kN
    force_unit: str = "kN"
    num_predictions: int
    inference_time_ms: float



# ================================================================
# TESTING FUNCTION
# ================================================================

def test_fuzzy_controller():
    """Test the fixed fuzzy controller"""
    
    print("\n" + "="*60)
    print("TESTING FIXED FUZZY CONTROLLER")
    print("="*60 + "\n")
    
    controller = FixedFuzzyTMDController()
    
    # Test cases: [relative_disp, relative_vel, expected_sign]
    test_cases = [
        (0.15, 0.8, -1, "TMD right + moving right → push LEFT"),
        (-0.15, -0.8, 1, "TMD left + moving left → push RIGHT"),
        (0.15, -0.5, -1, "TMD right + moving left → small push LEFT"),
        (-0.15, 0.5, 1, "TMD left + moving right → small push RIGHT"),
        (0.0, 0.0, 0, "At rest → near zero"),
        (0.2, 1.5, -1, "Far right + fast right → HARD push LEFT"),
        (-0.2, -1.5, 1, "Far left + fast left → HARD push RIGHT"),
    ]
    
    print(f"{'Rel Disp (m)':<15} {'Rel Vel (m/s)':<15} {'Force (kN)':<12} {'Expected':<10} {'Status':<10} Description")
    print("-" * 100)
    
    all_correct = True
    
    for disp, vel, expected_sign, description in test_cases:
        force_N = controller.compute(disp, vel)
        force_kN = force_N / 1000
        
        # Check sign
        if expected_sign == 0:
            correct = abs(force_kN) < 10  # Should be small
        else:
            correct = np.sign(force_kN) == expected_sign
        
        status = "✅ PASS" if correct else "❌ FAIL"
        if not correct:
            all_correct = False
        
        print(f"{disp:<15.3f} {vel:<15.3f} {force_kN:<12.2f} {expected_sign:<10} {status:<10} {description}")
    
    print("\n" + "="*60)
    if all_correct:
        print("✅ ALL TESTS PASSED! Controller working correctly.")
    else:
        print("❌ SOME TESTS FAILED! Review fuzzy rules.")
    print("="*60 + "\n")
    
    return controller


if __name__ == "__main__":
    # Run tests
    controller = test_fuzzy_controller()
    
    print("\n📋 NEXT STEPS:")
    print("1. Add this controller to your FastAPI application")
    print("2. Update the /fuzzylogic-batch endpoint")
    print("3. Deploy to Cloud Run")
    print("4. Test from MATLAB with relative motion inputs\n")
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

        # ================================================================
        # PRE-COMPUTE LOOKUP TABLE FOR FAST INTERPOLATION
        # ================================================================
        print("   FUZZY: Loading/building lookup table for fast simulation...")
        self._build_lookup_table()
        print(f"   FUZZY: Lookup table ready ({self.lut_disp_grid.shape[0]}x{self.lut_vel_grid.shape[0]} grid)")


    def _build_lookup_table(self):
        """
        Pre-compute fuzzy control surface as a lookup table for fast interpolation.
        This provides 50-100x speedup compared to calling fuzzy inference 6001 times.

        Uses caching to avoid recomputing the table on every startup (saves ~2 minutes).
        """
        import pickle
        from pathlib import Path

        # Cache file path (in same directory as this file)
        cache_dir = Path(__file__).parent
        cache_file = cache_dir / 'fuzzy_lookup_table_cache.pkl'

        # Try to load from cache
        if cache_file.exists():
            try:
                print("      FUZZY: Found cached lookup table, loading...")
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)

                self.lut_disp_grid = cache_data['disp_grid']
                self.lut_vel_grid = cache_data['vel_grid']
                self.lut_force = cache_data['force']

                print(f"      FUZZY: ✅ Loaded cached lookup table ({self.lut_force.shape[0]}x{self.lut_force.shape[1]} points)")
                return
            except Exception as e:
                print(f"      FUZZY: ⚠️  Cache load failed ({e}), rebuilding...")

        # Cache miss or load failed - compute from scratch
        print("      FUZZY: No cache found, computing lookup table (this takes ~2 minutes)...")

        # Create grid (fine enough for accurate interpolation)
        n_disp = 101  # -0.5 to 0.5 m in 0.01 m steps
        n_vel = 201   # -2.0 to 2.0 m/s in 0.02 m/s steps

        self.lut_disp_grid = np.linspace(-0.5, 0.5, n_disp)
        self.lut_vel_grid = np.linspace(-2.0, 2.0, n_vel)

        # Pre-compute force for each grid point
        self.lut_force = np.zeros((n_disp, n_vel))

        for i, disp in enumerate(self.lut_disp_grid):
            for j, vel in enumerate(self.lut_vel_grid):
                # Use actual fuzzy inference for grid points
                if abs(disp) < 1e-6 and abs(vel) < 1e-6:
                    self.lut_force[i, j] = 0.0
                else:
                    try:
                        self.controller.input['displacement'] = disp
                        self.controller.input['velocity'] = vel
                        self.controller.compute()
                        force = float(self.controller.output['force'])
                        self.lut_force[i, j] = np.clip(force, -100000, 100000)
                    except:
                        # Fallback to PD control
                        Kp, Kd = 50000, 20000
                        self.lut_force[i, j] = -Kp * disp - Kd * vel

        # Save to cache for next time
        try:
            print("      FUZZY: Saving lookup table to cache...")
            cache_data = {
                'disp_grid': self.lut_disp_grid,
                'vel_grid': self.lut_vel_grid,
                'force': self.lut_force
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"      FUZZY: ✅ Cached to {cache_file}")
        except Exception as e:
            print(f"      FUZZY: ⚠️  Failed to save cache ({e}), will recompute next time")


    def _interpolate_force(self, disp, vel):
        """
        Fast bilinear interpolation from pre-computed lookup table.
        50-100x faster than fuzzy inference.
        """
        # Clamp to grid bounds
        disp = np.clip(disp, -0.5, 0.5)
        vel = np.clip(vel, -2.0, 2.0)

        # Find grid indices
        i = np.searchsorted(self.lut_disp_grid, disp)
        j = np.searchsorted(self.lut_vel_grid, vel)

        # Handle edge cases
        if i == 0:
            i = 1
        if i >= len(self.lut_disp_grid):
            i = len(self.lut_disp_grid) - 1
        if j == 0:
            j = 1
        if j >= len(self.lut_vel_grid):
            j = len(self.lut_vel_grid) - 1

        # Bilinear interpolation
        d0, d1 = self.lut_disp_grid[i-1], self.lut_disp_grid[i]
        v0, v1 = self.lut_vel_grid[j-1], self.lut_vel_grid[j]

        # Interpolation weights
        wd = (disp - d0) / (d1 - d0) if d1 != d0 else 0
        wv = (vel - v0) / (v1 - v0) if v1 != v0 else 0

        # Bilinear interpolation formula
        f00 = self.lut_force[i-1, j-1]
        f01 = self.lut_force[i-1, j]
        f10 = self.lut_force[i, j-1]
        f11 = self.lut_force[i, j]

        force = (1-wd)*(1-wv)*f00 + (1-wd)*wv*f01 + wd*(1-wv)*f10 + wd*wv*f11

        return force


    def compute(self, roof_disp, roof_vel, tmd_disp, tmd_vel):
        """
        Compute control force for given absolute states (FAST VERSION with lookup table)

        Args:
            roof_disp: Roof displacement (meters)
            roof_vel: Roof velocity (m/s)
            tmd_disp: TMD displacement (meters)
            tmd_vel: TMD velocity (m/s)

        Returns:
            control_force: Force in Newtons (will be applied with Newton's 3rd law)
        """
        # Calculate relative states (TMD relative to roof)
        relative_displacement = tmd_disp - roof_disp
        relative_velocity = tmd_vel - roof_vel

        # Use fast interpolation from lookup table (50-100x faster than fuzzy inference!)
        force_N = self._interpolate_force(relative_displacement, relative_velocity)

        return force_N
    
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

    def simulate_episode(self, earthquake_data, dt=0.02):
        """
        Run closed-loop simulation with Fuzzy controller

        Args:
            earthquake_data: Ground acceleration time series (m/s²)
            dt: Time step (seconds)

        Returns:
            Dictionary with metrics (peak_roof, max_drift, DCR, forces, etc.)
        """
        # Initialize 12-story building + TMD (matching RL environment)
        n_floors = 12
        n_dof = 13  # 12 floors + 1 TMD

        # Building parameters (from RL environment)
        floor_mass = 1.0e6  # kg
        floor_stiffness = 5.0e8  # N/m
        zeta = 0.05  # 5% damping

        # TMD parameters
        tmd_mass = 1.2e5  # kg (120 tons)
        omega_1 = 3.14  # First mode frequency (rad/s)
        mu = tmd_mass / floor_mass
        omega_tmd = omega_1 / (1 + mu)  # Den Hartog tuning
        zeta_tmd = np.sqrt(3 * mu / (8 * (1 + mu)))
        tmd_k = tmd_mass * omega_tmd**2
        tmd_c = 2 * zeta_tmd * np.sqrt(tmd_k * tmd_mass)

        # Mass matrix
        M = np.eye(n_dof) * floor_mass
        M[12, 12] = tmd_mass

        # Stiffness matrix (tridiagonal for building)
        K = np.zeros((n_dof, n_dof))
        for i in range(n_floors):
            K[i, i] = 2 * floor_stiffness if i < n_floors - 1 else floor_stiffness
            if i > 0:
                K[i, i-1] = -floor_stiffness
                K[i-1, i] = -floor_stiffness

        # TMD coupling (roof is floor 11, TMD is index 12)
        roof_idx = 11
        K[roof_idx, roof_idx] += tmd_k
        K[roof_idx, 12] = -tmd_k
        K[12, roof_idx] = -tmd_k
        K[12, 12] = tmd_k

        # Damping matrix (Rayleigh damping)
        omega_1 = np.sqrt(np.linalg.eigvalsh(np.linalg.solve(M, K))[0])
        omega_2 = np.sqrt(np.linalg.eigvalsh(np.linalg.solve(M, K))[1])
        A = np.array([[1/omega_1, omega_1], [1/omega_2, omega_2]])
        coeffs = np.linalg.solve(A, [zeta, zeta])
        alpha, beta = coeffs
        C = alpha * M + beta * K

        # TMD damping
        C[roof_idx, roof_idx] += tmd_c
        C[roof_idx, 12] = -tmd_c
        C[12, roof_idx] = -tmd_c
        C[12, 12] = tmd_c

        # Newmark-beta parameters
        beta_newmark = 0.25
        gamma = 0.5

        # Pre-compute effective stiffness matrix and factorize it ONCE (HUGE performance gain!)
        K_eff = K + gamma / (beta_newmark * dt) * C + 1 / (beta_newmark * dt**2) * M

        # LU factorization (do this ONCE instead of 6001 times!)
        from scipy.linalg import lu_factor, lu_solve
        K_eff_factored = lu_factor(K_eff)

        # Pre-compute constant coefficients
        coeff_d = 1 / (beta_newmark * dt**2)
        coeff_c = gamma / (beta_newmark * dt)
        half_minus_beta = 0.5 - beta_newmark
        one_minus_gamma = 1 - gamma

        # Initialize state vectors
        displacement = np.zeros(n_dof)
        velocity = np.zeros(n_dof)
        acceleration = np.zeros(n_dof)

        # Pre-allocate tracking arrays (much faster than lists!)
        n_steps = len(earthquake_data)
        displacement_history = np.zeros(n_steps)  # Roof displacement only
        force_history = np.zeros(n_steps)
        drift_history = np.zeros((n_steps, n_floors))

        # Additional tracking for analysis plots
        peak_disp_by_floor = np.zeros(n_floors)  # Track peak displacement at each floor
        roof_accel_history = np.zeros(n_steps)   # Track roof acceleration

        # Pre-allocate earthquake force vector base (avoid repeated concatenation)
        eq_force_base = np.concatenate([np.ones(n_floors), [0]])

        # Simulation loop (closed-loop control!)
        for step in range(n_steps):
            # Get current ground acceleration
            ag = earthquake_data[step]

            # ============ CLOSED-LOOP CONTROL ============
            # Observe CURRENT state
            roof_disp = displacement[roof_idx]
            roof_vel = velocity[roof_idx]
            tmd_disp = displacement[12]
            tmd_vel = velocity[12]

            # Compute control force based on CURRENT state
            control_force = self.compute(roof_disp, roof_vel, tmd_disp, tmd_vel)
            force_history[step] = control_force

            # ============ PHYSICS SIMULATION ============
            # Earthquake force (optimized - reuse base vector)
            F_eq = -ag * M @ eq_force_base

            # Apply control force (Newton's 3rd law)
            F_eq[roof_idx] -= control_force  # Roof gets reaction
            F_eq[12] += control_force  # TMD gets actuator force

            # Newmark-beta time integration (optimized with pre-computed coefficients)
            d_pred = displacement + dt * velocity + half_minus_beta * dt**2 * acceleration
            v_pred = velocity + one_minus_gamma * dt * acceleration

            F_eff = F_eq + M @ (coeff_d * d_pred) + C @ (coeff_c * d_pred)

            # Use pre-factorized matrix (FAST!)
            displacement_new = lu_solve(K_eff_factored, F_eff)
            acceleration_new = coeff_d * (displacement_new - d_pred)
            velocity_new = v_pred + gamma * dt * acceleration_new

            # Update state
            displacement = displacement_new
            velocity = velocity_new
            acceleration = acceleration_new

            # Track metrics (optimized - direct array assignment)
            displacement_history[step] = displacement[roof_idx]
            roof_accel_history[step] = acceleration[roof_idx]

            # Update peak displacement for each floor
            for floor in range(n_floors):
                peak_disp_by_floor[floor] = max(peak_disp_by_floor[floor], abs(displacement[floor]))

            # Compute interstory drifts (vectorized where possible)
            drift_history[step, 0] = abs(displacement[0])
            drift_history[step, 1:] = np.abs(displacement[1:n_floors] - displacement[0:n_floors-1])

        # Compute metrics (arrays already pre-allocated, no conversion needed!)
        # 1. RMS of roof displacement
        rms_roof = np.sqrt(np.mean(displacement_history**2))

        # 2. Peak roof displacement
        peak_roof = np.max(np.abs(displacement_history))

        # 3. Max drift across all floors and time
        max_drift = np.max(drift_history)

        # 4. DCR (Drift Concentration Ratio)
        max_drift_per_floor = np.max(drift_history, axis=0)
        if len(max_drift_per_floor) > 0 and np.mean(max_drift_per_floor) > 1e-10:
            DCR = np.max(max_drift_per_floor) / np.mean(max_drift_per_floor)
        else:
            DCR = 0.0

        # 5. Peak and mean force
        peak_force = np.max(np.abs(force_history))
        mean_force = np.mean(np.abs(force_history))

        # 6. RMS roof acceleration
        rms_roof_accel = np.sqrt(np.mean(roof_accel_history**2))

        return {
            'rms_roof_displacement': float(rms_roof),
            'peak_roof_displacement': float(peak_roof),
            'max_drift': float(max_drift),
            'DCR': float(DCR),
            'peak_force': float(peak_force),
            'mean_force': float(mean_force),
            'peak_force_kN': float(peak_force / 1000),
            'mean_force_kN': float(mean_force / 1000),
            'forces': force_history.tolist(),
            'forces_N': force_history.tolist(),
            'forces_kN': (force_history / 1000).tolist(),
            'peak_disp_by_floor': peak_disp_by_floor.tolist(),  # NEW: Peak displacement at each floor
            'rms_roof_accel': float(rms_roof_accel)             # NEW: RMS roof acceleration
        }

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


class FuzzySimulationRequest(BaseModel):
    """Request for Fuzzy simulation endpoint"""
    earthquake_data: List[float] = Field(..., description="Ground acceleration time series (m/s²)")
    dt: float = Field(0.02, description="Time step (seconds)")


class FuzzySimulationResponse(BaseModel):
    """Response from Fuzzy simulation endpoint"""
    forces_N: List[float]
    forces_kN: List[float]
    count: int
    rms_roof_displacement: float
    peak_roof_displacement: float
    max_drift: float
    DCR: float
    peak_force: float
    mean_force: float
    peak_force_kN: float
    mean_force_kN: float
    model: str = "Fuzzy Logic Controller"
    simulation_time_ms: float



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
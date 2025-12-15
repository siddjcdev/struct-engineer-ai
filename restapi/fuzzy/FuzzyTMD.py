# ============================================================================
# COMPREHENSIVE FUZZY LOGIC TMD CONTROLLER
# ============================================================================
from skfuzzy import control as ctrl
import numpy as np
import skfuzzy as fuzz
from datetime import datetime
import json

class FuzzyTMDController:
    """
    Comprehensive Fuzzy Logic Controller for TMD
    Uses real physical values and engineering-based rules
    
    Input: displacement (m), velocity (m/s), acceleration (m/s²)
    Output: control force (N)
    """
    
    def __init__(self, 
                 displacement_range=(-0.5, 0.5),
                 velocity_range=(-2.0, 2.0),
                 force_range=(-100000, 100000)):
        """
        Initialize comprehensive fuzzy controller
        
        Args:
            displacement_range: (min, max) in meters
            velocity_range: (min, max) in m/s
            force_range: (min, max) in Newtons
        """
        self.displacement_range = displacement_range
        self.velocity_range = velocity_range
        self.force_range = force_range
        
        # Statistics tracking
        self.computation_count = 0
        self.last_computation_time = None
        self.computation_history = []
        
        # Build fuzzy system
        self.controller = self._build_fuzzy_system()
        
        print("="*70)
        print("COMPREHENSIVE FUZZY LOGIC TMD CONTROLLER INITIALIZED")
        print("="*70)
        print(f"Displacement range: {displacement_range[0]:.2f} to {displacement_range[1]:.2f} m")
        print(f"Velocity range: {velocity_range[0]:.2f} to {velocity_range[1]:.2f} m/s")
        print(f"Force range: {force_range[0]/1000:.1f} to {force_range[1]/1000:.1f} kN")
        print("="*70)
    
    def _build_fuzzy_system(self):
        """Build the comprehensive fuzzy inference system"""
        
        # Input variables
        displacement = ctrl.Antecedent(
            np.arange(self.displacement_range[0], self.displacement_range[1], 0.01),
            'displacement'
        )
        velocity = ctrl.Antecedent(
            np.arange(self.velocity_range[0], self.velocity_range[1], 0.01),
            'velocity'
        )
        
        # Output variable
        control_force = ctrl.Consequent(
            np.arange(self.force_range[0], self.force_range[1], 1000),
            'control_force'
        )
        
        # Membership functions - Displacement (5 levels for precision)
        displacement['negative_large'] = fuzz.trapmf(
            displacement.universe, 
            [self.displacement_range[0], self.displacement_range[0], -0.3, -0.1]
        )
        displacement['negative_small'] = fuzz.trimf(
            displacement.universe, [-0.3, -0.1, 0]
        )
        displacement['zero'] = fuzz.trimf(
            displacement.universe, [-0.1, 0, 0.1]
        )
        displacement['positive_small'] = fuzz.trimf(
            displacement.universe, [0, 0.1, 0.3]
        )
        displacement['positive_large'] = fuzz.trapmf(
            displacement.universe,
            [0.1, 0.3, self.displacement_range[1], self.displacement_range[1]]
        )
        
        # Membership functions - Velocity (5 levels for precision)
        velocity['negative_fast'] = fuzz.trapmf(
            velocity.universe,
            [self.velocity_range[0], self.velocity_range[0], -1.0, -0.3]
        )
        velocity['negative_slow'] = fuzz.trimf(
            velocity.universe, [-1.0, -0.3, 0]
        )
        velocity['zero'] = fuzz.trimf(
            velocity.universe, [-0.3, 0, 0.3]
        )
        velocity['positive_slow'] = fuzz.trimf(
            velocity.universe, [0, 0.3, 1.0]
        )
        velocity['positive_fast'] = fuzz.trapmf(
            velocity.universe,
            [0.3, 1.0, self.velocity_range[1], self.velocity_range[1]]
        )
        
        # Membership functions - Control Force (5 levels for precision)
        force_max = self.force_range[1]
        control_force['large_negative'] = fuzz.trapmf(
            control_force.universe,
            [self.force_range[0], self.force_range[0], -0.6*force_max, -0.2*force_max]
        )
        control_force['small_negative'] = fuzz.trimf(
            control_force.universe, [-0.6*force_max, -0.2*force_max, 0]
        )
        control_force['zero'] = fuzz.trimf(
            control_force.universe, [-0.2*force_max, 0, 0.2*force_max]
        )
        control_force['small_positive'] = fuzz.trimf(
            control_force.universe, [0, 0.2*force_max, 0.6*force_max]
        )
        control_force['large_positive'] = fuzz.trapmf(
            control_force.universe,
            [0.2*force_max, 0.6*force_max, self.force_range[1], self.force_range[1]]
        )
        
        # Comprehensive Fuzzy Rules (Engineering-based for TMD control)
        rules = [
            # Rule 1-2: Strong damping when moving fast outward
            ctrl.Rule(
                displacement['positive_large'] & velocity['positive_fast'],
                control_force['large_negative']
            ),
            ctrl.Rule(
                displacement['negative_large'] & velocity['negative_fast'],
                control_force['large_positive']
            ),
            
            # Rule 3-6: Moderate damping for moderate motion
            ctrl.Rule(
                displacement['positive_small'] & velocity['positive_slow'],
                control_force['small_negative']
            ),
            ctrl.Rule(
                displacement['positive_large'] & velocity['positive_slow'],
                control_force['small_negative']
            ),
            ctrl.Rule(
                displacement['negative_small'] & velocity['negative_slow'],
                control_force['small_positive']
            ),
            ctrl.Rule(
                displacement['negative_large'] & velocity['negative_slow'],
                control_force['small_positive']
            ),
            
            # Rule 7: Minimal force near equilibrium
            ctrl.Rule(
                displacement['zero'] & velocity['zero'],
                control_force['zero']
            ),
            
            # Rule 8-11: Reduced damping when naturally returning to equilibrium
            ctrl.Rule(
                displacement['positive_small'] & velocity['negative_slow'],
                control_force['zero']
            ),
            ctrl.Rule(
                displacement['positive_large'] & velocity['negative_fast'],
                control_force['small_positive']
            ),
            ctrl.Rule(
                displacement['negative_small'] & velocity['positive_slow'],
                control_force['zero']
            ),
            ctrl.Rule(
                displacement['negative_large'] & velocity['positive_fast'],
                control_force['small_negative']
            ),
        ]
        
        # Create control system
        control_system = ctrl.ControlSystem(rules)
        print(f"✅ Fuzzy system built with {len(rules)} engineering-based rules")
        
        return ctrl.ControlSystemSimulation(control_system)
    
    def compute(self, displacement, velocity, acceleration=None):
        """
        Compute control force for given building state
        
        Args:
            displacement: Inter-story drift in meters
            velocity: Inter-story drift velocity in m/s
            acceleration: Building acceleration in m/s² (optional, for logging)
        
        Returns:
            control_force: Control force in Newtons
        """
        self.computation_count += 1
        self.last_computation_time = datetime.now()
        
        # Clip to valid ranges
        displacement_clipped = np.clip(
            displacement,
            self.displacement_range[0],
            self.displacement_range[1]
        )
        velocity_clipped = np.clip(
            velocity,
            self.velocity_range[0],
            self.velocity_range[1]
        )
        
        # Set inputs
        self.controller.input['displacement'] = displacement_clipped
        self.controller.input['velocity'] = velocity_clipped
        
        # Compute
        try:
            self.controller.compute()
            control_force = self.controller.output['control_force']
            
            # Store computation history
            computation_record = {
                'timestamp': self.last_computation_time.isoformat(),
                'count': self.computation_count,
                'inputs': {
                    'displacement': displacement,
                    'displacement_clipped': displacement_clipped,
                    'velocity': velocity,
                    'velocity_clipped': velocity_clipped,
                    'acceleration': acceleration
                },
                'output': {
                    'control_force_N': control_force,
                    'control_force_kN': control_force / 1000
                }
            }
            
            # Keep last 1000 computations
            if len(self.computation_history) >= 1000:
                self.computation_history.pop(0)
            self.computation_history.append(computation_record)
            
            return control_force
            
        except Exception as e:
            print(f"❌ Fuzzy computation error: {e}")
            return 0.0
    
    def compute_batch(self, displacements, velocities, accelerations=None):
        """
        Compute control forces for time series data
        
        Args:
            displacements: Array of inter-story drifts (m)
            velocities: Array of velocities (m/s)
            accelerations: Array of accelerations (m/s²), optional
        
        Returns:
            forces: Array of control forces (N)
        """
        forces = []
        
        if accelerations is None:
            accelerations = [None] * len(displacements)
        
        for d, v, a in zip(displacements, velocities, accelerations):
            forces.append(self.compute(d, v, a))
        
        return np.array(forces)
    
    def get_stats(self):
        """Get controller statistics"""
        return {
            "total_computations": self.computation_count,
            "last_computation": self.last_computation_time.isoformat() if self.last_computation_time else None,
            "displacement_range_m": self.displacement_range,
            "velocity_range_ms": self.velocity_range,
            "force_range_kN": [self.force_range[0]/1000, self.force_range[1]/1000],
            "status": "active"
        }
    
    def save_computation_history(self, filepath):
        """Save computation history to JSON file"""
        with open(filepath, 'w') as f:
            json.dump({
                'controller_config': {
                    'displacement_range': self.displacement_range,
                    'velocity_range': self.velocity_range,
                    'force_range': self.force_range
                },
                'total_computations': self.computation_count,
                'computation_history': self.computation_history
            }, f, indent=2)
        return filepath

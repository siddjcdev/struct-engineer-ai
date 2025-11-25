"""
Fuzzy Logic TMD Controller - Standalone Implementation
Pure fuzzy logic controller for Tuned Mass Damper (TMD) systems
No API calls or network dependencies required

Usage:
    from fuzzy_tmd_controller import FuzzyTMDController
    
    controller = FuzzyTMDController()
    force = controller.compute(displacement=0.2, velocity=0.5)
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


class FuzzyTMDController:
    """
    Pure fuzzy logic controller for TMD
    Input: displacement (m), velocity (m/s)
    Output: control force (N)
    """
    
    def __init__(self, 
                 displacement_range=(-0.5, 0.5),
                 velocity_range=(-2.0, 2.0),
                 force_range=(-100000, 100000)):
        """
        Initialize fuzzy controller
        
        Args:
            displacement_range: (min, max) in meters
            velocity_range: (min, max) in m/s
            force_range: (min, max) in Newtons
        """
        self.displacement_range = displacement_range
        self.velocity_range = velocity_range
        self.force_range = force_range
        
        self.controller = self._build_fuzzy_system()
    
    def _build_fuzzy_system(self):
        """Build the fuzzy inference system"""
        
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
        
        # Membership functions - Displacement
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
        
        # Membership functions - Velocity
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
        
        # Membership functions - Control Force
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
        
        # Fuzzy Rules (Engineering-based)
        rules = [
            # Strong damping when moving fast outward
            ctrl.Rule(
                displacement['positive_large'] & velocity['positive_fast'],
                control_force['large_negative']
            ),
            ctrl.Rule(
                displacement['negative_large'] & velocity['negative_fast'],
                control_force['large_positive']
            ),
            
            # Moderate damping for moderate motion
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
            
            # Minimal force near equilibrium
            ctrl.Rule(
                displacement['zero'] & velocity['zero'],
                control_force['zero']
            ),
            
            # Less damping when naturally returning to equilibrium
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
        return ctrl.ControlSystemSimulation(control_system)
    
    def compute(self, displacement, velocity):
        """
        Compute control force for given state
        
        Args:
            displacement: Building displacement in meters
            velocity: Building velocity in m/s
        
        Returns:
            control_force: Control force in Newtons
        """
        # Clip to valid ranges
        displacement = np.clip(
            displacement,
            self.displacement_range[0],
            self.displacement_range[1]
        )
        velocity = np.clip(
            velocity,
            self.velocity_range[0],
            self.velocity_range[1]
        )
        
        # Set inputs
        self.controller.input['displacement'] = displacement
        self.controller.input['velocity'] = velocity
        
        # Compute
        try:
            self.controller.compute()
            return self.controller.output['control_force']
        except:
            # If computation fails, return zero force
            return 0.0
    
    def compute_batch(self, displacements, velocities):
        """
        Compute control forces for time series
        
        Args:
            displacements: Array of displacements (m)
            velocities: Array of velocities (m/s)
        
        Returns:
            forces: Array of control forces (N)
        """
        forces = []
        for d, v in zip(displacements, velocities):
            forces.append(self.compute(d, v))
        return np.array(forces)


# Simple usage example
if __name__ == "__main__":
    print("="*60)
    print("Fuzzy TMD Controller - Pure Logic Module")
    print("="*60)
    
    # Create controller
    controller = FuzzyTMDController()
    
    # Test single values
    print("\nTesting single computations:")
    test_cases = [
        (0.2, 0.5, "Positive displacement, positive velocity"),
        (-0.2, -0.5, "Negative displacement, negative velocity"),
        (0.0, 0.0, "Equilibrium"),
        (0.3, -0.3, "Positive displacement, returning"),
    ]
    
    for disp, vel, description in test_cases:
        force = controller.compute(disp, vel)
        print(f"\n{description}:")
        print(f"  Input:  d={disp:>6.2f}m, v={vel:>6.2f}m/s")
        print(f"  Output: F={force:>10.1f}N ({force/1000:>6.1f}kN)")
    
    # Test time series
    print("\n" + "="*60)
    print("Testing batch computation:")
    time = np.linspace(0, 10, 100)
    displacements = 0.1 * np.sin(2*np.pi*0.5*time)
    velocities = np.gradient(displacements, time[1]-time[0])
    
    forces = controller.compute_batch(displacements, velocities)
    
    print(f"Processed {len(time)} time steps")
    print(f"Max force: {np.max(np.abs(forces))/1000:.1f} kN")
    print(f"Mean force: {np.mean(np.abs(forces))/1000:.1f} kN")
    
    print("\n✅ Fuzzy controller ready to use!")
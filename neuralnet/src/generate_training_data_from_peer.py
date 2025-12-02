"""
Generate TMD Training Data from PEER Earthquakes
Simulates 12-story building with soft 8th floor responding to real earthquakes
Uses fuzzy logic controller to label optimal control forces
"""

import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Tuple
from scipy.integrate import odeint
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from datetime import datetime


class BuildingSimulator:
    """
    12-story building with soft 8th floor
    Matches your MATLAB model parameters
    """
    
    def __init__(self):
        # Building parameters (12-story structure)
        self.n_stories = 12
        self.story_height = 3.6  # meters
        self.m_floor = 500000  # kg per floor (500 tons)
        
        # Stiffness (N/m) - 8th floor is weakened
        self.k_normal = 1.5e8  # Normal story stiffness
        self.k_soft = 0.3e8    # Soft story (20% of normal)
        
        # Build stiffness vector
        self.k = np.ones(self.n_stories) * self.k_normal
        self.k[7] = self.k_soft  # 8th floor is soft (index 7)
        
        # Damping (use Rayleigh damping, 5% critical)
        self.zeta = 0.05
        
        # Mass matrix
        self.M = np.diag(np.ones(self.n_stories) * self.m_floor)
        
        # Stiffness matrix (tridiagonal)
        self.K = self._build_stiffness_matrix()
        
        # Damping matrix (Rayleigh: C = a0*M + a1*K)
        omega = np.sqrt(self.k[0] / self.m_floor)  # Fundamental frequency
        self.C = 2 * self.zeta * omega * self.M
        
        # TMD parameters (on roof)
        self.m_tmd = 0.02 * self.n_stories * self.m_floor  # 2% of total mass
        self.tmd_location = self.n_stories - 1  # Roof (index 11)
        
        print(f"Building initialized:")
        print(f"  Stories: {self.n_stories}")
        print(f"  Mass per floor: {self.m_floor/1000:.0f} tons")
        print(f"  Soft story: 8th floor (k = {self.k_soft/1e8:.1f}e8 N/m)")
        print(f"  TMD mass: {self.m_tmd/1000:.0f} tons (2% of total)")
    
    def _build_stiffness_matrix(self) -> np.ndarray:
        """Build stiffness matrix for building"""
        K = np.zeros((self.n_stories, self.n_stories))
        
        # First floor
        K[0, 0] = self.k[0] + self.k[1]
        K[0, 1] = -self.k[1]
        
        # Middle floors
        for i in range(1, self.n_stories - 1):
            K[i, i-1] = -self.k[i]
            K[i, i] = self.k[i] + self.k[i+1]
            K[i, i+1] = -self.k[i+1]
        
        # Top floor
        K[-1, -2] = -self.k[-1]
        K[-1, -1] = self.k[-1]
        
        return K
    
    def simulate_earthquake_response(
        self,
        earthquake_accel: np.ndarray,
        dt: float,
        control_forces: np.ndarray = None
    ) -> Dict[str, np.ndarray]:
        """
        Simulate building response to earthquake
        
        Args:
            earthquake_accel: Ground acceleration (m/s²)
            dt: Time step (s)
            control_forces: Optional TMD control forces at each timestep (N)
            
        Returns:
            Dictionary with displacement, velocity, acceleration histories
        """
        n_steps = len(earthquake_accel)
        t = np.arange(n_steps) * dt
        
        if control_forces is None:
            control_forces = np.zeros(n_steps)
        
        # State vector: [x1, x2, ..., x12, v1, v2, ..., v12]
        # where xi = displacement of floor i, vi = velocity of floor i
        state0 = np.zeros(2 * self.n_stories)
        
        # Define equations of motion
        def equations_of_motion(state, t_current):
            # Get current index
            idx = int(t_current / dt)
            if idx >= n_steps:
                idx = n_steps - 1
            
            # Extract displacements and velocities
            x = state[:self.n_stories]
            v = state[self.n_stories:]
            
            # External force from earthquake (acts on all masses)
            F_earthquake = -self.m_floor * earthquake_accel[idx] * np.ones(self.n_stories)
            
            # TMD control force (acts on roof)
            F_control = np.zeros(self.n_stories)
            F_control[self.tmd_location] = control_forces[idx]
            
            # Total external force
            F_ext = F_earthquake + F_control
            
            # Compute accelerations: M*a = F_ext - C*v - K*x
            a = np.linalg.solve(self.M, F_ext - self.C @ v - self.K @ x)
            
            # Return derivatives [v, a]
            return np.concatenate([v, a])
        
        # Solve ODE
        solution = odeint(equations_of_motion, state0, t)
        
        # Extract results
        displacement = solution[:, :self.n_stories].T  # Shape: (12, n_steps)
        velocity = solution[:, self.n_stories:].T
        
        # Compute accelerations
        acceleration = np.zeros_like(displacement)
        for i in range(n_steps):
            x = displacement[:, i]
            v = velocity[:, i]
            F_earthquake = -self.m_floor * earthquake_accel[i] * np.ones(self.n_stories)
            F_control = np.zeros(self.n_stories)
            F_control[self.tmd_location] = control_forces[i]
            F_ext = F_earthquake + F_control
            acceleration[:, i] = np.linalg.solve(self.M, F_ext - self.C @ v - self.K @ x)
        
        return {
            'time': t,
            'displacement': displacement,  # (12, n_steps)
            'velocity': velocity,
            'acceleration': acceleration,
            'control_force': control_forces
        }


class FuzzyTMDController:
    """
    Fuzzy logic controller for TMD
    Provides "ground truth" labels for neural network training
    """
    
    def __init__(self):
        # Input 1: Displacement (m)
        self.displacement = ctrl.Antecedent(np.linspace(-0.5, 0.5, 100), 'displacement')
        self.displacement['negative_large'] = fuzz.trimf(self.displacement.universe, [-0.5, -0.5, -0.2])
        self.displacement['negative_small'] = fuzz.trimf(self.displacement.universe, [-0.3, -0.1, 0])
        self.displacement['zero'] = fuzz.trimf(self.displacement.universe, [-0.1, 0, 0.1])
        self.displacement['positive_small'] = fuzz.trimf(self.displacement.universe, [0, 0.1, 0.3])
        self.displacement['positive_large'] = fuzz.trimf(self.displacement.universe, [0.2, 0.5, 0.5])
        
        # Input 2: Velocity (m/s)
        self.velocity = ctrl.Antecedent(np.linspace(-2.0, 2.0, 100), 'velocity')
        self.velocity['negative_large'] = fuzz.trimf(self.velocity.universe, [-2.0, -2.0, -0.8])
        self.velocity['negative_small'] = fuzz.trimf(self.velocity.universe, [-1.2, -0.4, 0])
        self.velocity['zero'] = fuzz.trimf(self.velocity.universe, [-0.4, 0, 0.4])
        self.velocity['positive_small'] = fuzz.trimf(self.velocity.universe, [0, 0.4, 1.2])
        self.velocity['positive_large'] = fuzz.trimf(self.velocity.universe, [0.8, 2.0, 2.0])
        
        # Output: Control force (kN)
        self.force = ctrl.Consequent(np.linspace(-100, 100, 200), 'force')
        self.force['negative_large'] = fuzz.trimf(self.force.universe, [-100, -100, -40])
        self.force['negative_small'] = fuzz.trimf(self.force.universe, [-60, -20, 0])
        self.force['zero'] = fuzz.trimf(self.force.universe, [-20, 0, 20])
        self.force['positive_small'] = fuzz.trimf(self.force.universe, [0, 20, 60])
        self.force['positive_large'] = fuzz.trimf(self.force.universe, [40, 100, 100])
        
        # Define control rules
        rules = [
            # When moving right (positive displacement)
            ctrl.Rule(self.displacement['positive_large'] & self.velocity['positive_large'], 
                     self.force['negative_large']),
            ctrl.Rule(self.displacement['positive_large'] & self.velocity['positive_small'], 
                     self.force['negative_large']),
            ctrl.Rule(self.displacement['positive_small'] & self.velocity['positive_large'], 
                     self.force['negative_small']),
            ctrl.Rule(self.displacement['positive_small'] & self.velocity['positive_small'], 
                     self.force['negative_small']),
            
            # When moving left (negative displacement)
            ctrl.Rule(self.displacement['negative_large'] & self.velocity['negative_large'], 
                     self.force['positive_large']),
            ctrl.Rule(self.displacement['negative_large'] & self.velocity['negative_small'], 
                     self.force['positive_large']),
            ctrl.Rule(self.displacement['negative_small'] & self.velocity['negative_large'], 
                     self.force['positive_small']),
            ctrl.Rule(self.displacement['negative_small'] & self.velocity['negative_small'], 
                     self.force['positive_small']),
            
            # Near zero - damping based on velocity
            ctrl.Rule(self.displacement['zero'] & self.velocity['positive_large'], 
                     self.force['negative_small']),
            ctrl.Rule(self.displacement['zero'] & self.velocity['positive_small'], 
                     self.force['negative_small']),
            ctrl.Rule(self.displacement['zero'] & self.velocity['zero'], 
                     self.force['zero']),
            ctrl.Rule(self.displacement['zero'] & self.velocity['negative_small'], 
                     self.force['positive_small']),
            ctrl.Rule(self.displacement['zero'] & self.velocity['negative_large'], 
                     self.force['positive_small']),
        ]
        
        # Create control system
        self.control_system = ctrl.ControlSystem(rules)
        self.controller = ctrl.ControlSystemSimulation(self.control_system)
        
        print("Fuzzy controller initialized with 13 rules")
    
    def compute(self, displacement: float, velocity: float) -> float:
        """
        Compute optimal control force
        
        Args:
            displacement: Building displacement (m)
            velocity: Building velocity (m/s)
            
        Returns:
            Control force (kN)
        """
        # Clip inputs to valid range to prevent fuzzy controller failures
        displacement = np.clip(displacement, -0.5, 0.5)
        velocity = np.clip(velocity, -2.0, 2.0)
        
        try:
            # Compute fuzzy output
            self.controller.input['displacement'] = displacement
            self.controller.input['velocity'] = velocity
            self.controller.compute()
            
            force_kN = self.controller.output['force']
            
            # Check for invalid outputs
            if np.isnan(force_kN) or np.isinf(force_kN):
                # Fallback to simple proportional control
                force_kN = -40 * displacement - 10 * velocity
                
            return force_kN
            
        except Exception as e:
            # If fuzzy controller fails, use simple proportional-derivative control
            # This ensures we always return a valid control force
            force_kN = -40 * displacement - 10 * velocity
            return force_kN


class TrainingDataGenerator:
    """Generate training data from PEER earthquakes"""
    
    def __init__(self):
        self.building = BuildingSimulator()
        self.fuzzy_controller = FuzzyTMDController()
    
    def generate_training_samples(
        self,
        earthquakes: List[Dict],
        use_fuzzy_labels: bool = True
    ) -> List[Tuple[float, float, float]]:
        """
        Generate training samples from earthquake dataset
        
        Args:
            earthquakes: List of earthquake dictionaries
            use_fuzzy_labels: If True, use fuzzy controller for labels
            
        Returns:
            List of (displacement, velocity, force) tuples
        """
        print("="*70)
        print("GENERATING TRAINING DATA")
        print("="*70)
        print()
        
        training_data = []
        
        for i, eq in enumerate(earthquakes):
            print(f"Processing earthquake {i+1}/{len(earthquakes)}: {eq['id']}")
            print(f"  Magnitude: {eq['magnitude']:.1f}, PGA: {eq['pga_g']:.3f}g, Duration: {eq['duration']:.1f}s")
            
            # Get earthquake data
            accel = np.array(eq['acceleration'])
            dt = eq['dt']
            n_steps = len(accel)
            
            # Simulate without control to get building states
            response = self.building.simulate_earthquake_response(accel, dt)
            
            # Use roof displacement and velocity (most critical for TMD)
            roof_disp = response['displacement'][-1, :]  # Last floor (roof)
            roof_vel = response['velocity'][-1, :]
            
            # Generate control forces using fuzzy logic
            if use_fuzzy_labels:
                control_forces = np.array([
                    self.fuzzy_controller.compute(d, v) * 1000  # Convert kN to N
                    for d, v in zip(roof_disp, roof_vel)
                ])
            else:
                # Passive case - no control
                control_forces = np.zeros(n_steps)
            
            # Create training samples
            # Skip first 100 steps (initial transient)
            skip = min(100, n_steps // 10)
            for j in range(skip, n_steps):
                training_data.append((
                    float(roof_disp[j]),          # Input: displacement (m)
                    float(roof_vel[j]),           # Input: velocity (m/s)
                    float(control_forces[j]/1000) # Output: force (kN)
                ))
            
            n_samples = n_steps - skip
            print(f"  Generated {n_samples} training samples")
            print(f"  Total samples so far: {len(training_data)}")
            print()
        
        print("="*70)
        print(f"✅ TRAINING DATA GENERATION COMPLETE")
        print(f"Total samples: {len(training_data)}")
        print("="*70)
        print()
        
        return training_data
    
    def save_training_data(self, training_data: List[Tuple], filename: str = 'tmd_training_data.json'):
        """Save training data to JSON"""
        data_dict = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'n_samples': len(training_data),
                'input_features': ['displacement_m', 'velocity_ms'],
                'output_feature': 'control_force_kN',
                'building': {
                    'n_stories': self.building.n_stories,
                    'soft_floor': 8,
                    'tmd_location': 'roof'
                }
            },
            'samples': [
                {
                    'displacement': d,
                    'velocity': v,
                    'force': f
                }
                for d, v, f in training_data
            ]
        }
        
        filepath = Path(filename)
        with open(filepath, 'w') as f:
            json.dump(data_dict, f, indent=2)
        
        print(f"✅ Saved training data to {filepath}")
        print(f"   File size: {filepath.stat().st_size / 1024 / 1024:.1f} MB")
        
        return filepath


def main():
    """Main execution"""
    print("="*70)
    print("TMD TRAINING DATA GENERATION FROM PEER EARTHQUAKES")
    print("="*70)
    print()
    
    # Load earthquake dataset
    print("Loading PEER earthquake dataset...")
    with open('peer_earthquake_data/peer_earthquake_dataset.json', 'r') as f:
        earthquakes = json.load(f)
    print(f"  ✅ Loaded {len(earthquakes)} earthquakes")
    print()
    
    # Generate training data
    generator = TrainingDataGenerator()
    training_data = generator.generate_training_samples(earthquakes, use_fuzzy_labels=True)
    
    # Save to file
    output_path = generator.save_training_data(training_data, 'tmd_training_data_peer.json')
    
    # Show statistics
    displacements = [d for d, v, f in training_data]
    velocities = [v for d, v, f in training_data]
    forces = [f for d, v, f in training_data]
    
    print()
    print("Training Data Statistics:")
    print(f"  Samples: {len(training_data)}")
    print(f"  Displacement range: [{min(displacements):.3f}, {max(displacements):.3f}] m")
    print(f"  Velocity range: [{min(velocities):.3f}, {max(velocities):.3f}] m/s")
    print(f"  Force range: [{min(forces):.1f}, {max(forces):.1f}] kN")
    print()
    
    print("="*70)
    print("✅ READY FOR NEURAL NETWORK TRAINING")
    print("="*70)
    print()
    print("Next step:")
    print("  python train_neural_network_peer.py")


if __name__ == '__main__':
    main()
    
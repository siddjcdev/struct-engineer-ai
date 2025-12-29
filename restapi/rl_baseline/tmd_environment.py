"""
TMD BUILDING ENVIRONMENT FOR REINFORCEMENT LEARNING
===================================================

Custom Gymnasium environment for training RL agents to control a TMD system.

The agent learns to minimize building displacement by controlling the TMD force.

Author: Siddharth
Date: December 2025
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple


class TMDBuildingEnv(gym.Env):
    """
    Gymnasium Environment for TMD Building Control
    
    State: [roof_disp, roof_vel, tmd_disp, tmd_vel]
    Action: Control force in range [-1, 1] (scaled to ±100kN)
    Reward: -|roof_displacement| (negative = minimize displacement)
    """
                                             
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        earthquake_data: np.ndarray,
        dt: float = 0.02,
        max_force: float = 200000.0,  # 1)150 2)200, kN in Newtons
        earthquake_name: str = "Unknown"
    ):
        """
        Initialize the TMD building environment
        
        Args:
            earthquake_data: Ground acceleration time series (m/s²)
            dt: Time step (seconds)
            max_force: Maximum control force (Newtons)
            earthquake_name: Name for logging/debugging
        """
        super().__init__()
        
        self.earthquake_data = earthquake_data
        self.dt = dt
        self.max_force = max_force
        self.earthquake_name = earthquake_name
        
        # Building parameters (same as MATLAB)
        self.n_floors = 12
        self.floor_mass = 300000  # kg
        self.tmd_mass = 0.02 * self.floor_mass  # 2% mass ratio
        self.tmd_k = 50e3  # N/m
        self.tmd_c = 2000  # N·s/m (passive damping)
        
        # Story stiffness
        k_typical = 800e6  # N/m
        self.story_stiffness = k_typical * np.ones(self.n_floors)
        self.story_stiffness[7] = 0.5 * k_typical  # Soft 8th floor (index 7)
        
        self.damping_ratio = 0.02  # 2%
        
        # Build system matrices
        self.M = self._build_mass_matrix()
        self.K = self._build_stiffness_matrix()
        self.C = self._build_damping_matrix()
        
        # Newmark integration parameters
        self.beta = 0.25
        self.gamma = 0.5
        
        # State: [roof_disp, roof_vel, tmd_disp, tmd_vel]
        # Using realistic bounds from earthquake simulations
        self.observation_space = spaces.Box(
            low=np.array([-0.5, -2.0, -0.6, -2.5]),   # min values
            high=np.array([0.5, 2.0, 0.6, 2.5]),      # max values
            dtype=np.float32
        )
        
        # Action: normalized force in [-1, 1]
        # Will be scaled to [-max_force, +max_force]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )
        
        # Episode tracking
        self.current_step = 0
        self.max_steps = len(earthquake_data)
        
        # State vectors (13 DOFs: 12 floors + TMD)
        self.displacement = np.zeros(13)
        self.velocity = np.zeros(13)
        self.acceleration = np.zeros(13)
        
        # Performance tracking
        self.peak_displacement = 0.0
        self.cumulative_displacement = 0.0

        # NEW: Track additional metrics for API responses
        self.displacement_history = []  # For RMS calculation
        self.force_history = []  # For peak and mean force
        self.drift_history = []  # For max drift and DCR
        
    
    def _build_mass_matrix(self) -> np.ndarray:
        """Build mass matrix (13x13)"""
        M = np.zeros((13, 13))
        for i in range(self.n_floors):
            M[i, i] = self.floor_mass
        M[12, 12] = self.tmd_mass
        return M
    
    
    def _build_stiffness_matrix(self) -> np.ndarray:
        """Build stiffness matrix (13x13)"""
        K = np.zeros((13, 13))
        
        # Building stiffness
        for i in range(self.n_floors):
            K[i, i] += self.story_stiffness[i]
            if i < self.n_floors - 1:
                K[i, i] += self.story_stiffness[i + 1]
                K[i, i + 1] = -self.story_stiffness[i + 1]
                K[i + 1, i] = -self.story_stiffness[i + 1]
        
        # TMD coupling
        roof_idx = self.n_floors - 1  # Index 11
        K[roof_idx, roof_idx] += self.tmd_k
        K[roof_idx, 12] = -self.tmd_k
        K[12, roof_idx] = -self.tmd_k
        K[12, 12] = self.tmd_k
        
        return K
    
    
    def _build_damping_matrix(self) -> np.ndarray:
        """Build Rayleigh damping matrix"""
        # Compute eigenvalues
        eigenvalues = np.linalg.eigvals(np.linalg.solve(self.M, self.K))
        # Use small positive threshold for numerical stability
        omega = np.sqrt(np.real(eigenvalues[eigenvalues > 1e-10]))
        omega = np.sort(omega)

        if len(omega) < 2:
            raise ValueError(f"System has fewer than 2 positive eigenvalues: {len(omega)}. Check mass/stiffness matrices.")

        omega1 = omega[0]
        omega2 = omega[1]
        zeta = self.damping_ratio
        
        # Rayleigh damping coefficients
        A = np.array([[1/(2*omega1), omega1/2],
                      [1/(2*omega2), omega2/2]])
        coeffs = np.linalg.solve(A, [zeta, zeta])
        alpha, beta = coeffs
        
        C = alpha * self.M + beta * self.K
        
        # Add TMD damping
        roof_idx = self.n_floors - 1
        C[roof_idx, roof_idx] += self.tmd_c
        C[roof_idx, 12] -= self.tmd_c
        C[12, roof_idx] -= self.tmd_c
        C[12, 12] += self.tmd_c
        
        return C
    
    
    def _newmark_step(
        self,
        d: np.ndarray,
        v: np.ndarray,
        a: np.ndarray,
        F: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Newmark-beta time integration"""
        
        d_pred = d + self.dt * v + (0.5 - self.beta) * self.dt**2 * a
        v_pred = v + (1 - self.gamma) * self.dt * a
        
        K_eff = (self.K + 
                 self.gamma / (self.beta * self.dt) * self.C + 
                 1 / (self.beta * self.dt**2) * self.M)
        
        F_eff = (F + 
                 self.M @ (1 / (self.beta * self.dt**2) * d_pred) + 
                 self.C @ (self.gamma / (self.beta * self.dt) * d_pred))
        
        d_new = np.linalg.solve(K_eff, F_eff)
        a_new = (1 / (self.beta * self.dt**2)) * (d_new - d_pred)
        v_new = v_pred + self.gamma * self.dt * a_new
        
        return d_new, v_new, a_new
    
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        """Reset environment to initial state"""
        
        super().reset(seed=seed)
        
        self.current_step = 0
        self.displacement = np.zeros(13)
        self.velocity = np.zeros(13)
        self.acceleration = np.zeros(13)
        
        self.peak_displacement = 0.0
        self.cumulative_displacement = 0.0

        # Clear metric histories
        self.displacement_history = []
        self.force_history = []
        self.drift_history = []

        # Initial observation
        obs = np.array([
            self.displacement[11],  # roof displacement
            self.velocity[11],      # roof velocity
            self.displacement[12],  # TMD displacement
            self.velocity[12]       # TMD velocity
        ], dtype=np.float32)
        
        info = {
            'earthquake': self.earthquake_name,
            'timestep': 0
        }
        
        return obs, info
    
    
    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one timestep
        
        Args:
            action: Normalized force [-1, 1]
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        
        # Scale action to actual force
        control_force = float(action[0]) * self.max_force
        
        # Get ground acceleration for this timestep
        ag = self.earthquake_data[self.current_step]
        
        # Earthquake force on building
        F_eq = -ag * self.M @ np.concatenate([np.ones(self.n_floors), [0]])
        
        # Apply control force with Newton's 3rd law
        F_eq[11] -= control_force  # Roof (reaction)
        F_eq[12] += control_force  # TMD (action)
        
        # Time integration
        self.displacement, self.velocity, self.acceleration = self._newmark_step(
            self.displacement,
            self.velocity,
            self.acceleration,
            F_eq
        )
        
        # Extract roof state
        roof_disp = self.displacement[11]
        roof_vel = self.velocity[11]
        tmd_disp = self.displacement[12]
        tmd_vel = self.velocity[12]
        
        # Observation
        obs = np.array([roof_disp, roof_vel, tmd_disp, tmd_vel], dtype=np.float32)
        
        # Reward: negative absolute displacement (minimize displacement)
        reward = -abs(roof_disp)
        
        # Bonus penalty for excessive control force (energy efficiency)
        energy_penalty = -0.0001 * (control_force / self.max_force)**2
        reward += energy_penalty
        
        # Update tracking
        self.peak_displacement = max(self.peak_displacement, abs(roof_disp))
        self.cumulative_displacement += abs(roof_disp)

        # NEW: Track metrics for final reporting
        self.displacement_history.append(roof_disp)
        self.force_history.append(control_force)

        # Compute interstory drifts for all floors
        drifts = self._compute_interstory_drifts(self.displacement[:self.n_floors])
        self.drift_history.append(drifts)

        # Episode termination
        self.current_step += 1
        terminated = False  # Episode runs to completion
        truncated = self.current_step >= self.max_steps
        
        # Info
        info = {
            'timestep': self.current_step,
            'roof_displacement': roof_disp,
            'control_force': control_force,
            'peak_displacement': self.peak_displacement,
            'cumulative_displacement': self.cumulative_displacement
        }
        
        return obs, reward, terminated, truncated, info


    def _compute_interstory_drifts(self, displacements: np.ndarray) -> np.ndarray:
        """
        Compute interstory drifts for all floors

        Args:
            displacements: Floor displacements [floor_0, floor_1, ..., floor_N-1]

        Returns:
            drifts: Interstory drifts [drift_1, drift_2, ..., drift_N]
        """
        n = len(displacements)
        drifts = np.zeros(n)

        # First floor drift (relative to ground)
        drifts[0] = abs(displacements[0])

        # Other floors (relative to floor below)
        for i in range(1, n):
            drifts[i] = abs(displacements[i] - displacements[i-1])

        return drifts


    def get_episode_metrics(self) -> dict:
        """
        Calculate and return all episode metrics

        Returns:
            dict with:
                - rms_roof_displacement: RMS of roof displacement (m)
                - peak_roof_displacement: Peak roof displacement (m)
                - max_drift: Maximum interstory drift across all floors and timesteps (m)
                - DCR: Drift Concentration Ratio (dimensionless)
                - peak_force: Peak control force (N)
                - mean_force: Mean absolute control force (N)
        """
        if len(self.displacement_history) == 0:
            return {
                'rms_roof_displacement': 0.0,
                'peak_roof_displacement': 0.0,
                'max_drift': 0.0,
                'DCR': 0.0,
                'peak_force': 0.0,
                'mean_force': 0.0
            }

        # Convert to numpy arrays
        displacements = np.array(self.displacement_history)
        forces = np.array(self.force_history)

        # 1. RMS of roof displacement
        rms_roof = np.sqrt(np.mean(displacements**2))

        # 2. Peak roof displacement
        peak_roof = np.max(np.abs(displacements))

        # 3. Max drift across all floors and time
        drift_array = np.array(self.drift_history)  # Shape: (timesteps, n_floors)
        max_drift = np.max(drift_array)

        # 4. DCR (Drift Concentration Ratio)
        # For each floor, get its max drift over time
        max_drift_per_floor = np.max(np.abs(drift_array), axis=0)  # Shape: (n_floors,)

        if len(max_drift_per_floor) > 0:
            sorted_peaks = np.sort(max_drift_per_floor)
            percentile_75 = np.percentile(sorted_peaks, 75)
            max_peak = np.max(max_drift_per_floor)

            if percentile_75 > 1e-10:
                DCR = max_peak / percentile_75
            else:
                DCR = 0.0
        else:
            DCR = 0.0

        # 5. Peak and mean force
        peak_force = np.max(np.abs(forces))
        mean_force = np.mean(np.abs(forces))

        return {
            'rms_roof_displacement': float(rms_roof),
            'peak_roof_displacement': float(peak_roof),
            'max_drift': float(max_drift),
            'DCR': float(DCR),
            'peak_force': float(peak_force),
            'mean_force': float(mean_force),
            'peak_force_kN': float(peak_force / 1000),
            'mean_force_kN': float(mean_force / 1000)
        }


    def render(self):
        """Optional rendering (not implemented)"""
        pass


# ================================================================
# HELPER FUNCTION TO CREATE ENVIRONMENT
# ================================================================

def make_tmd_env(
    earthquake_file: str,
    earthquake_name: str = None
) -> TMDBuildingEnv:
    """
    Factory function to create TMD environment from earthquake file
    
    Args:
        earthquake_file: Path to CSV file with earthquake data
        earthquake_name: Optional name for the earthquake
        
    Returns:
        TMDBuildingEnv instance
    """
    
    # Load earthquake data
    data = np.loadtxt(earthquake_file, delimiter=',', skiprows=1)
    
    # Extract time and acceleration
    if data.shape[1] >= 2:
        times = data[:, 0]
        accelerations = data[:, 1]
        dt = np.mean(np.diff(times))
    else:
        accelerations = data
        dt = 0.02
    
    # Get name from filename if not provided
    if earthquake_name is None:
        import os
        earthquake_name = os.path.basename(earthquake_file)
    
    return TMDBuildingEnv(
        earthquake_data=accelerations,
        dt=dt,
        earthquake_name=earthquake_name
    )


if __name__ == "__main__":
    # Quick test
    print("Testing TMD Environment...")
    
    # Create a simple test earthquake
    t = np.linspace(0, 20, 1000)
    test_earthquake = 3.0 * np.sin(2 * np.pi * 1.5 * t) * np.exp(-0.1 * t)
    
    env = TMDBuildingEnv(
        earthquake_data=test_earthquake,
        dt=0.02,
        earthquake_name="Test Earthquake"
    )
    
    print(f"✅ Environment created")
    print(f"   Observation space: {env.observation_space}")
    print(f"   Action space: {env.action_space}")
    print(f"   Episode length: {env.max_steps} steps")
    
    # Test reset
    obs, info = env.reset()
    print(f"\n✅ Reset successful")
    print(f"   Initial observation: {obs}")
    
    # Test a few steps
    print(f"\n✅ Testing 5 steps...")
    for i in range(5):
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"   Step {i+1}: reward={reward:.6f}, roof_disp={obs[0]:.6f}m")
    
    print(f"\n✅ Environment test complete!")

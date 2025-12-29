"""
IMPROVED TMD ENVIRONMENT - ALL FIXES APPLIED
============================================

Improvements:
1. Multi-objective reward function (displacement + velocity + force + acceleration)
2. Force smoothness regularization
3. Shorter episode length (matches earthquake duration)
4. Better state normalization

Author: Siddharth
Date: December 2025
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple


class ImprovedTMDBuildingEnv(gym.Env):
    """
    Improved TMD Building Environment with better reward function
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        earthquake_data: np.ndarray,
        dt: float = 0.02,
        max_force: float = 150000.0,  # Start with 50 kN (will use curriculum)
        earthquake_name: str = "Unknown"
    ):
        super().__init__()
        
        self.earthquake_data = earthquake_data
        self.dt = dt
        self.max_force = max_force
        self.earthquake_name = earthquake_name
        
        # Building parameters
        self.n_floors = 12
        self.floor_mass = 300000
        self.tmd_mass = 0.02 * self.floor_mass
        self.tmd_k = 50e3
        self.tmd_c = 2000
        
        k_typical = 800e6
        self.story_stiffness = k_typical * np.ones(self.n_floors)
        self.story_stiffness[7] = 0.5 * k_typical
        
        self.damping_ratio = 0.02
        
        # Build matrices
        self.M = self._build_mass_matrix()
        self.K = self._build_stiffness_matrix()
        self.C = self._build_damping_matrix()
        
        # Newmark parameters
        self.beta = 0.25
        self.gamma = 0.5
        
        # State space
        self.observation_space = spaces.Box(
            low=np.array([-0.5, -2.0, -0.6, -2.5]),
            high=np.array([0.5, 2.0, 0.6, 2.5]),
            dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )
        
        # Episode length - IMPROVED: Match earthquake duration
        self.current_step = 0
        self.max_steps = min(len(earthquake_data), 2000)  # Max 40 seconds
        
        # State vectors
        self.displacement = np.zeros(13)
        self.velocity = np.zeros(13)
        self.acceleration = np.zeros(13)
        
        # Performance tracking
        self.peak_displacement = 0.0
        self.cumulative_displacement = 0.0

        # IMPROVEMENT: Track previous force for smoothness penalty
        self.previous_force = 0.0

        # IMPROVEMENT: Track roof acceleration for comfort
        self.roof_acceleration = 0.0

        # NEW: Track additional metrics for API responses
        self.displacement_history = []  # For RMS calculation
        self.force_history = []  # For peak and mean force
        self.drift_history = []  # For max drift and DCR
    
    
    def _build_mass_matrix(self) -> np.ndarray:
        M = np.zeros((13, 13))
        for i in range(self.n_floors):
            M[i, i] = self.floor_mass
        M[12, 12] = self.tmd_mass
        return M
    
    
    def _build_stiffness_matrix(self) -> np.ndarray:
        K = np.zeros((13, 13))
        for i in range(self.n_floors):
            K[i, i] += self.story_stiffness[i]
            if i < self.n_floors - 1:
                K[i, i] += self.story_stiffness[i + 1]
                K[i, i + 1] = -self.story_stiffness[i + 1]
                K[i + 1, i] = -self.story_stiffness[i + 1]
        
        roof_idx = self.n_floors - 1
        K[roof_idx, roof_idx] += self.tmd_k
        K[roof_idx, 12] = -self.tmd_k
        K[12, roof_idx] = -self.tmd_k
        K[12, 12] = self.tmd_k
        
        return K
    
    
    def _build_damping_matrix(self) -> np.ndarray:
        eigenvalues = np.linalg.eigvals(np.linalg.solve(self.M, self.K))
        omega = np.sqrt(np.real(eigenvalues[eigenvalues > 0]))
        omega = np.sort(omega)
        
        omega1 = omega[0]
        omega2 = omega[1]
        zeta = self.damping_ratio
        
        A = np.array([[1/(2*omega1), omega1/2],
                      [1/(2*omega2), omega2/2]])
        coeffs = np.linalg.solve(A, [zeta, zeta])
        alpha, beta = coeffs
        
        C = alpha * self.M + beta * self.K
        
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
        
        super().reset(seed=seed)
        
        self.current_step = 0
        self.displacement = np.zeros(13)
        self.velocity = np.zeros(13)
        self.acceleration = np.zeros(13)
        
        self.peak_displacement = 0.0
        self.cumulative_displacement = 0.0
        self.previous_force = 0.0
        self.roof_acceleration = 0.0

        # Clear metric histories
        self.displacement_history = []
        self.force_history = []
        self.drift_history = []

        obs = np.array([
            self.displacement[11],
            self.velocity[11],
            self.displacement[12],
            self.velocity[12]
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
        
        # Scale action to actual force
        control_force = float(action[0]) * self.max_force
        
        # Get ground acceleration
        ag = self.earthquake_data[self.current_step]
        
        # Earthquake force
        F_eq = -ag * self.M @ np.concatenate([np.ones(self.n_floors), [0]])
        
        # Apply control with Newton's 3rd law
        F_eq[11] -= control_force
        F_eq[12] += control_force
        
        # Time integration
        self.displacement, self.velocity, self.acceleration = self._newmark_step(
            self.displacement,
            self.velocity,
            self.acceleration,
            F_eq
        )
        
        # Extract state
        roof_disp = self.displacement[11]
        roof_vel = self.velocity[11]
        tmd_disp = self.displacement[12]
        tmd_vel = self.velocity[12]
        
        # IMPROVEMENT: Track roof acceleration
        self.roof_acceleration = self.acceleration[11]
        
        # Observation
        obs = np.array([roof_disp, roof_vel, tmd_disp, tmd_vel], dtype=np.float32)
        
        # ================================================================
        # IMPROVED MULTI-OBJECTIVE REWARD FUNCTION
        # ================================================================
        
        # 1. Primary objective: Minimize displacement
        displacement_penalty = -1.0 * abs(roof_disp)
        
        # 2. Dampen oscillations: Penalize velocity
        velocity_penalty = -0.3 * abs(roof_vel)
        
        # 3. Energy efficiency: Penalize large forces
        force_normalized = control_force / self.max_force
        #force_penalty = -0.01 * (force_normalized ** 2)
        #IMPROVEMENT: No force penalty
        force_penalty = 0.0  # Don't penalize force usage at all

        # 4. Smoothness: Penalize rapid force changes
        force_change = abs(control_force - self.previous_force)
        smoothness_penalty = -0.005 * (force_change / self.max_force)
        
        # 5. Comfort: Penalize high accelerations
        acceleration_penalty = -0.1 * abs(self.roof_acceleration)

        # 6. Drift distribution: Penalize drift concentration (DCR)
        # Calculate interstory drifts from current floor displacements
        floor_displacements = self.displacement[:self.n_floors]
        floor_drifts = np.diff(floor_displacements)  # Interstory drifts

        if len(floor_drifts) > 0:
            abs_drifts = np.abs(floor_drifts)
            sorted_drifts = np.sort(abs_drifts)
            percentile_75 = np.percentile(sorted_drifts, 75)
            max_drift = np.max(abs_drifts)

            if percentile_75 > 1e-10:
                instantaneous_dcr = max_drift / percentile_75
                # Penalize deviation from ideal DCR=1.0
                # Weight at 0.3x to encourage uniform drift without dominating displacement objective
                dcr_penalty = -0.3 * (instantaneous_dcr - 1.0)
            else:
                dcr_penalty = 0.0
        else:
            dcr_penalty = 0.0

        # Combined reward
        reward = (
            displacement_penalty +
            velocity_penalty +
            force_penalty +
            smoothness_penalty +
            acceleration_penalty +
            dcr_penalty
        )
        #IMPROVEMENT: Simplified reward to just displacement
        #reward = -abs(roof_disp)

        # Update previous force for next step
        self.previous_force = control_force
        
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
        terminated = False
        truncated = self.current_step >= self.max_steps
        
        # Info
        info = {
            'timestep': self.current_step,
            'roof_displacement': roof_disp,
            'roof_velocity': roof_vel,
            'roof_acceleration': self.roof_acceleration,
            'control_force': control_force,
            'peak_displacement': self.peak_displacement,
            'cumulative_displacement': self.cumulative_displacement,
            'reward_breakdown': {
                'displacement': displacement_penalty,
                'velocity': velocity_penalty,
                'force': force_penalty,
                'smoothness': smoothness_penalty,
                'acceleration': acceleration_penalty
            }
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
        max_drift_per_floor = np.max(drift_array, axis=0)  # Shape: (n_floors,)

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
        pass


# ================================================================
# FACTORY FUNCTION
# ================================================================

def make_improved_tmd_env(
    earthquake_file: str,
    earthquake_name: str = None,
    max_force: float = 150000.0
) -> ImprovedTMDBuildingEnv:
    """Create improved TMD environment"""
    
    print(f"Loading earthquake data from {earthquake_file}...")
    data = np.loadtxt(earthquake_file, delimiter=',', skiprows=1)
    print(f"✅ Earthquake data loaded: {data.shape[0]} samples")

    if data.shape[1] >= 2:
        times = data[:, 0]
        accelerations = data[:, 1]
        dt = np.mean(np.diff(times))
    else:
        accelerations = data
        dt = 0.02
    
    if earthquake_name is None:
        import os
        earthquake_name = os.path.basename(earthquake_file)
    
    return ImprovedTMDBuildingEnv(
        earthquake_data=accelerations,
        dt=dt,
        max_force=max_force,
        earthquake_name=earthquake_name
    )


if __name__ == "__main__":
    print("Testing Improved TMD Environment...")
    
    # Test with synthetic earthquake
    t = np.linspace(0, 20, 1000)
    test_earthquake = 3.0 * np.sin(2 * np.pi * 1.5 * t) * np.exp(-0.1 * t)
    
    env = ImprovedTMDBuildingEnv(
        earthquake_data=test_earthquake,
        dt=0.02,
        earthquake_name="Test"
    )
    
    print(f"✅ Improved environment created")
    print(f"   Episode length: {env.max_steps} steps (vs 6000 in old version)")
    print(f"   Force limit: {env.max_force/1000:.0f} kN")
    
    # Test episode
    obs, info = env.reset()
    total_reward = 0
    
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        if i == 0:
            print(f"\n✅ Reward breakdown (first step):")
            for key, value in info['reward_breakdown'].items():
                print(f"   {key}: {value:.6f}")
    
    print(f"\n✅ Test complete! Total reward: {total_reward:.2f}")
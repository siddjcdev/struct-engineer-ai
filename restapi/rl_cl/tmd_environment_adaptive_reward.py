"""
TMD ENVIRONMENT v7 - ADAPTIVE REWARD SCALING
=============================================

v7 IMPROVEMENTS:
- Magnitude-adaptive reward scaling
- Automatically adjusts reward signal based on earthquake intensity
- Prevents gradient instability on extreme earthquakes
- Optimal scaling per magnitude class

Scaling Strategy:
- Small earthquakes (M4.5): 3× multiplier (less aggressive)
- Moderate earthquakes (M5.7): 7× multiplier (strong signal)
- High earthquakes (M7.4): 4× multiplier (balanced)
- Extreme earthquakes (M8.4): 3× multiplier (conservative)

Author: Siddharth
Date: January 2026
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
        earthquake_name: str = "Unknown",
        sensor_noise_std: float = 0.0,
        actuator_noise_std: float = 0.0,
        latency_steps: int = 0,
        dropout_prob: float = 0.0,
        obs_bounds: dict = None,
        reward_scale: float = None  # NEW: Adaptive reward scaling (auto-computed if None)
    ):
        super().__init__()

        self.earthquake_data = earthquake_data
        self.dt = dt
        self.max_force = max_force
        self.earthquake_name = earthquake_name

        # NEW: Compute adaptive reward scaling based on earthquake intensity
        if reward_scale is None:
            # Auto-detect magnitude from earthquake name or compute from PGA
            self.reward_scale = self._compute_adaptive_reward_scale()
        else:
            self.reward_scale = reward_scale

        # Domain randomization parameters
        self.sensor_noise_std = sensor_noise_std
        self.actuator_noise_std = actuator_noise_std
        self.latency_steps = latency_steps
        self.dropout_prob = dropout_prob

        # Building parameters - UPDATED to match MATLAB configuration
        # This ensures consistent results between MATLAB and Python simulations
        self.n_floors = 12
        self.floor_mass = 2.0e5          # 200,000 kg (was 300,000) - matches MATLAB m0
        self.tmd_mass = 0.02 * self.floor_mass  # 2% mass ratio = 4000 kg

        # TMD tuning - OPTIMIZED for active control (FIX #1)
        # Building fundamental frequency: 0.193 Hz
        # CRITICAL: For active control with 50-150 kN forces, TMD must be stiffer
        # Passive-optimized TMD (k=3765) caused runaway displacement (867-6780 cm!)
        # Active-optimized TMD prevents runaway and allows proper control
        # Trade-off: Passive performance is 0% (but was only 3.5% anyway)
        # Benefit: Active control can work properly without observation clipping
        self.tmd_k = 50000               # TMD stiffness (50 kN/m) - active control optimized
        self.tmd_c = 2000                # TMD damping (2000 N·s/m) - active control optimized

        # Story stiffness - matches MATLAB k0 and soft_story_factor
        k_typical = 2.0e7                # 20 MN/m (was 800 MN/m) - matches MATLAB k0
        self.story_stiffness = k_typical * np.ones(self.n_floors)
        self.story_stiffness[7] = 0.60 * k_typical  # 60% stiffness (was 50%) - matches MATLAB

        self.damping_ratio = 0.015       # 1.5% damping (was 2%) - matches MATLAB zeta_target

        # Build matrices
        self.M = self._build_mass_matrix()
        self.K = self._build_stiffness_matrix()
        self.C = self._build_damping_matrix()

        # Newmark parameters
        self.beta = 0.25
        self.gamma = 0.5

        # State: [roof_disp, roof_vel, floor8_disp, floor8_vel, floor6_disp, floor6_vel, tmd_disp, tmd_vel]
        # UPDATED: Expanded observation space to include mid-floors for better DCR control
        # - Roof (floor 12): Global response
        # - Floor 8: Weak floor (critical for DCR)
        # - Floor 6: Mid-height reference
        # - TMD: Active control device
        # Bounds: Adaptive based on earthquake magnitude (larger bounds for extreme events)
        # Default: ±1.2m displacement, ±3.0m/s velocity (for M4.5-M5.7)
        # Can be overridden for M7.4+ earthquakes
        if obs_bounds is None:
            obs_bounds = {
                'disp': 5.0, 'vel': 20.0, 'tmd_disp': 15.0, 'tmd_vel': 60.0
            }
        self.observation_space = spaces.Box(
            low=np.array([
                -obs_bounds['disp'], -obs_bounds['vel'],
                -obs_bounds['disp'], -obs_bounds['vel'],
                -obs_bounds['disp'], -obs_bounds['vel'],
                -obs_bounds['tmd_disp'], -obs_bounds['tmd_vel']
            ]),
            high=np.array([
                obs_bounds['disp'], obs_bounds['vel'],
                obs_bounds['disp'], obs_bounds['vel'],
                obs_bounds['disp'], obs_bounds['vel'],
                obs_bounds['tmd_disp'], obs_bounds['tmd_vel']
            ]),
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )

        # Episode length - FIXED: Use full earthquake duration
        # CRITICAL: Train on full duration so model learns to handle entire earthquake
        # Training: 40s (2000 steps), Test: 60s-120s (3000-6000 steps)
        self.current_step = 0
        self.max_steps = len(earthquake_data)  # Full duration, no artificial limit!

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

        # Track peak drift per floor for consistent DCR calculation
        self.peak_drift_per_floor = np.zeros(self.n_floors)

        # Domain randomization: Action buffer for latency
        self.action_buffer = []


    def _compute_adaptive_reward_scale(self) -> float:
        """
        Compute adaptive reward scaling based on earthquake characteristics

        Strategy based on v5/v6 results:
        - M4.5: Use 3× (gentle - avoid over-penalizing small displacements)
        - M5.7: Use 7× (strong - showed best improvement with strong signal)
        - M7.4: Use 4× (balanced - 10× caused instability, 5× marginal)
        - M8.4: Use 3× (conservative - extreme earthquakes need gentler signal)

        Returns reward multiplier
        """
        # Compute PGA from earthquake data
        pga_g = np.max(np.abs(self.earthquake_data)) / 9.81

        # Magnitude-adaptive scaling based on PGA
        if pga_g < 0.30:  # M4.5 range (PGA ~0.25g)
            scale = 3.0
            magnitude = "M4.5"
        elif pga_g < 0.55:  # M5.7 range (PGA ~0.35g)
            scale = 7.0
            magnitude = "M5.7"
        elif pga_g < 0.85:  # M7.4 range (PGA ~0.75g)
            scale = 4.0
            magnitude = "M7.4"
        else:  # M8.4 range (PGA ~0.9g)
            scale = 3.0
            magnitude = "M8.4"

        print(f"   Adaptive reward scale: {scale}× for PGA={pga_g:.3f}g (detected as {magnitude})")
        return scale


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
        # Use small positive threshold for numerical stability
        omega = np.sqrt(np.real(eigenvalues[eigenvalues > 1e-10]))
        omega = np.sort(omega)

        if len(omega) < 2:
            raise ValueError(f"System has fewer than 2 positive eigenvalues: {len(omega)}. Check mass/stiffness matrices.")

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

        # Clear peak drift tracking
        self.peak_drift_per_floor = np.zeros(self.n_floors)

        # Clear action buffer for latency
        self.action_buffer = []

        # Expanded observation: roof, floor8, floor6, TMD
        obs = np.array([
            self.displacement[11],  # roof displacement
            self.velocity[11],      # roof velocity
            self.displacement[7],   # floor 8 displacement (weak floor)
            self.velocity[7],       # floor 8 velocity
            self.displacement[5],   # floor 6 displacement (mid-height)
            self.velocity[5],       # floor 6 velocity
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

        # ================================================================
        # DOMAIN RANDOMIZATION: Apply actuator noise and latency
        # ================================================================

        # 1. Add actuator noise to action
        if self.actuator_noise_std > 0:
            noise = np.random.normal(0, self.actuator_noise_std, size=action.shape)
            action_noisy = action + noise
            action_noisy = np.clip(action_noisy, -1.0, 1.0)  # Keep in valid range
        else:
            action_noisy = action

        # 2. Apply latency by buffering actions
        if self.latency_steps > 0:
            self.action_buffer.append(action_noisy)
            if len(self.action_buffer) > self.latency_steps:
                action_delayed = self.action_buffer.pop(0)
            else:
                # During warmup, use zero action
                action_delayed = np.zeros_like(action_noisy)
        else:
            action_delayed = action_noisy

        # Scale action to actual force
        control_force = float(action_delayed[0]) * self.max_force

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

        # Extract state (expanded: roof, floor8, floor6, TMD)
        roof_disp = self.displacement[11]
        roof_vel = self.velocity[11]
        floor8_disp = self.displacement[7]   # Weak floor
        floor8_vel = self.velocity[7]
        floor6_disp = self.displacement[5]   # Mid-height
        floor6_vel = self.velocity[5]
        tmd_disp = self.displacement[12]
        tmd_vel = self.velocity[12]

        # IMPROVEMENT: Track roof acceleration
        self.roof_acceleration = self.acceleration[11]

        # ================================================================
        # DOMAIN RANDOMIZATION: Apply sensor noise and dropout
        # ================================================================

        # 3. Add sensor noise to observations (expanded to 8 values)
        if self.sensor_noise_std > 0:
            sensor_noise = np.random.normal(0, self.sensor_noise_std, size=8)
            obs_noisy = np.array([
                roof_disp, roof_vel, floor8_disp, floor8_vel,
                floor6_disp, floor6_vel, tmd_disp, tmd_vel
            ], dtype=np.float32) * (1 + sensor_noise)
        else:
            obs_noisy = np.array([
                roof_disp, roof_vel, floor8_disp, floor8_vel,
                floor6_disp, floor6_vel, tmd_disp, tmd_vel
            ], dtype=np.float32)

        # 4. Apply sensor dropout
        if self.dropout_prob > 0:
            dropout_mask = np.random.random(8) > self.dropout_prob
            obs_noisy = obs_noisy * dropout_mask

        # Final observation
        obs = obs_noisy
        
        # ================================================================
        # SHAPED REWARD FUNCTION - Guides agent toward correct control
        # ================================================================

        # NEW INSIGHT: Force direction alone isn't enough - velocity-based switching
        # can actually pump energy into the system if timing is wrong!
        #
        # The emergency_physics_check showed constant +1.0 achieves 18.38cm,
        # but our velocity-based policy gets 24.94cm (worse than 21.02cm uncontrolled).
        #
        # Solution: Give agent SMALL displacement penalty as primary objective,
        # Remove force direction bonus (it was teaching wrong behavior).
        # Let agent discover the right control strategy through displacement minimization.

        # 1. Displacement: ADAPTIVE penalty based on earthquake magnitude (v7)
        displacement_penalty = -self.reward_scale * abs(roof_disp)  # Adaptive scaling!

        # 2. Velocity: ADAPTIVE penalty based on earthquake magnitude (v7)
        velocity_penalty = -(self.reward_scale * 0.3) * abs(roof_vel)  # Adaptive scaling!

        # 3. Energy efficiency: Disabled
        force_penalty = 0.0

        # 4. Smoothness: Disabled
        smoothness_penalty = 0.0

        # 5. Comfort: Disabled
        acceleration_penalty = 0.0

        # 6. Force direction bonus: REMOVED
        # Velocity-based switching was actually ADDING energy to the system
        # Let agent learn optimal control through displacement minimization alone
        force_direction_bonus = 0.0

        # 7. DCR tracking (for metrics only, NOT used in reward)
        # Track peak drift for each floor over time, then calculate DCR
        # DCR is still computed and logged but doesn't affect the reward signal
        # Hypothesis: Good vibration control naturally produces good DCR
        floor_drifts = self._compute_interstory_drifts(self.displacement[:self.n_floors])
        self.peak_drift_per_floor = np.maximum(self.peak_drift_per_floor, floor_drifts)

        # Combined reward - ADAPTIVE SIGNAL (v7)
        # Magnitude-adaptive penalties prevent gradient instability
        # No force direction shaping, let agent discover optimal control
        reward = (
            displacement_penalty +      # -scale * |disp| (adaptive!)
            velocity_penalty +          # -(scale*0.3) * |vel| (adaptive!)
            force_penalty +             # 0.0
            smoothness_penalty +        # 0.0
            acceleration_penalty +      # 0.0
            force_direction_bonus       # 0.0 (removed - was teaching wrong behavior)
            # NO dcr_penalty - let it emerge naturally (proven: DCR=0.00)
        )

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
                'acceleration': acceleration_penalty,
                'force_direction': force_direction_bonus,  # NEW: Track direction bonus
                'dcr': 0.0  # Not used in reward (tracking only)
            }
        }

        # Add episode-level metrics when episode ends (for TensorBoard logging)
        if truncated or terminated:
            episode_metrics = self.get_episode_metrics()
            info['max_isdr_percent'] = episode_metrics['max_isdr_percent']
            info['max_dcr'] = episode_metrics['DCR']
            info['peak_displacement'] = episode_metrics['peak_roof_displacement']

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
                - max_isdr: Maximum Interstory Drift Ratio (as decimal, e.g. 0.006 = 0.6%)
                - max_isdr_percent: Maximum ISDR as percentage (e.g. 0.6)
                - DCR: Drift Concentration Ratio (dimensionless)
                - peak_force: Peak control force (N)
                - mean_force: Mean absolute control force (N)
                - peak_force_kN: Peak control force (kN)
                - mean_force_kN: Mean absolute control force (kN)
        """
        if len(self.displacement_history) == 0:
            return {
                'rms_roof_displacement': 0.0,
                'peak_roof_displacement': 0.0,
                'max_drift': 0.0,
                'max_isdr': 0.0,
                'max_isdr_percent': 0.0,
                'DCR': 0.0,
                'peak_force': 0.0,
                'mean_force': 0.0,
                'peak_force_kN': 0.0,
                'mean_force_kN': 0.0
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

        # 4. ISDR (Interstory Drift Ratio)
        # Max drift divided by story height
        story_height = 3.6  # meters
        max_isdr = max_drift / story_height
        max_isdr_percent = max_isdr * 100  # Convert to percentage

        # 5. DCR (Drift Concentration Ratio)
        # For each floor, get its max drift over time
        max_drift_per_floor = np.max(np.abs(drift_array), axis=0)  # Shape: (n_floors,)

        if len(max_drift_per_floor) > 0:
            sorted_peaks = np.sort(max_drift_per_floor)
            percentile_75 = np.percentile(sorted_peaks, 75)
            max_peak = np.max(max_drift_per_floor)

            if percentile_75 > 0.001:  # 1mm minimum drift (prevents early-episode explosion)
                DCR = max_peak / percentile_75
            else:
                DCR = 0.0
        else:
            DCR = 0.0

        # 6. Peak and mean force
        peak_force = np.max(np.abs(forces))
        mean_force = np.mean(np.abs(forces))

        return {
            'rms_roof_displacement': float(rms_roof),
            'peak_roof_displacement': float(peak_roof),
            'max_drift': float(max_drift),
            'max_isdr': float(max_isdr),
            'max_isdr_percent': float(max_isdr_percent),
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
    max_force: float = 150000.0,
    sensor_noise_std: float = 0.0,
    actuator_noise_std: float = 0.0,
    latency_steps: int = 0,
    dropout_prob: float = 0.0,
    obs_bounds: dict = None,
    reward_scale: float = None
) -> ImprovedTMDBuildingEnv:
    """
    Create improved TMD environment with optional domain randomization

    Args:
        earthquake_file: Path to earthquake CSV file
        earthquake_name: Name of the earthquake (optional)
        max_force: Maximum control force in Newtons
        sensor_noise_std: Standard deviation of sensor noise (0.0 = no noise)
        actuator_noise_std: Standard deviation of actuator noise (0.0 = no noise)
        latency_steps: Number of timesteps of actuator latency (0 = no latency)
        dropout_prob: Probability of sensor dropout (0.0 = no dropout)
        obs_bounds: Custom observation bounds dict with keys 'disp', 'vel', 'tmd_disp', 'tmd_vel'
                   If None, uses default bounds (1.2m, 3.0m/s, 1.5m, 3.5m/s)
        reward_scale: Fixed reward scaling multiplier (None = auto-compute based on earthquake PGA)
                     Use 1.0 for consistent training across different magnitudes

    Returns:
        ImprovedTMDBuildingEnv instance with domain randomization
    """

    print(f"Loading earthquake data from {earthquake_file}...")
    data = np.loadtxt(earthquake_file, delimiter=',', skiprows=1)
    print(f"✅ Earthquake data loaded: {data.shape[0]} samples")

    if data.shape[1] >= 2:
        times = data[:, 0]
        accelerations = data[:, 1]
        dt = float(np.mean(np.diff(times)))
    else:
        accelerations = data
        dt = 0.02

    if earthquake_name is None:
        import os
        earthquake_name = os.path.basename(earthquake_file)

    # Display domain randomization settings
    if sensor_noise_std > 0 or actuator_noise_std > 0 or latency_steps > 0 or dropout_prob > 0:
        print(f"✓ Domain randomization enabled:")
        if sensor_noise_std > 0:
            print(f"   - Sensor noise: {sensor_noise_std*100:.1f}%")
        if actuator_noise_std > 0:
            print(f"   - Actuator noise: {actuator_noise_std*100:.1f}%")
        if latency_steps > 0:
            print(f"   - Latency: {latency_steps} steps ({latency_steps*dt*1000:.0f}ms)")
        if dropout_prob > 0:
            print(f"   - Dropout: {dropout_prob*100:.1f}%")

    # Display custom observation bounds if provided
    if obs_bounds is not None:
        print(f"✓ Custom observation bounds:")
        print(f"   - Displacement: ±{obs_bounds['disp']:.1f} m")
        print(f"   - Velocity: ±{obs_bounds['vel']:.1f} m/s")
        print(f"   - TMD displacement: ±{obs_bounds['tmd_disp']:.1f} m")
        print(f"   - TMD velocity: ±{obs_bounds['tmd_vel']:.1f} m/s")

    return ImprovedTMDBuildingEnv(
        earthquake_data=accelerations,
        dt=dt,
        max_force=max_force,
        earthquake_name=earthquake_name,
        sensor_noise_std=sensor_noise_std,
        actuator_noise_std=actuator_noise_std,
        latency_steps=latency_steps,
        dropout_prob=dropout_prob,
        obs_bounds=obs_bounds,
        reward_scale=reward_scale  # Pass through reward_scale (None = auto-compute)
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
"""
TMD ENVIRONMENT v12 - SOFT STORY TMD PLACEMENT
===============================================

v12 KEY CHANGES:
- TMD mounted AT FLOOR 8 (soft story) instead of roof
- Tuned to soft-story mode frequency
- 300 kN max force for aggressive control
- New reward function with specified target penalties
- Optimized for ISDR reduction at critical floor

This configuration directly addresses the weak floor vulnerability
by placing the TMD where it can most effectively control drift.

Author: Claude Code
Date: January 2026
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple


class SoftStoryTMDEnv(gym.Env):
    """
    TMD Environment with TMD placed at soft story (floor 8) for maximum ISDR control
    """

    metadata = {'render_modes': ['human']}

    def __init__(
        self,
        earthquake_data: np.ndarray,
        dt: float = 0.02,
        max_force: float = 300000.0,  # 300 kN for aggressive control
        earthquake_name: str = "Unknown",
        sensor_noise_std: float = 0.0,
        actuator_noise_std: float = 0.0,
        latency_steps: int = 0,
        dropout_prob: float = 0.0,
        obs_bounds: dict = None,
        reward_scale: float = 1.0  # Fixed reward scale
    ):
        super().__init__()

        self.earthquake_data = earthquake_data
        self.dt = dt
        self.max_force = max_force
        self.earthquake_name = earthquake_name
        self.reward_scale = reward_scale

        # Domain randomization parameters
        self.sensor_noise_std = sensor_noise_std
        self.actuator_noise_std = actuator_noise_std
        self.latency_steps = latency_steps
        self.dropout_prob = dropout_prob

        # Building parameters
        self.n_floors = 12
        self.floor_mass = 2.0e5  # 200,000 kg per floor

        # TMD configuration - MOUNTED AT FLOOR 8 (soft story)
        self.tmd_floor = 7  # Floor 8 (0-indexed as 7)
        self.tmd_mass = 0.04 * self.floor_mass  # 4% of floor mass = 8000 kg

        # TMD tuning - tuned to soft-story mode (will calculate after building stiffness matrix)
        # Placeholder values - will be updated in _tune_tmd_to_soft_story()
        self.tmd_k = 15000  # Will be optimized
        self.tmd_c = 800    # Will be optimized

        # Story stiffness with soft story at floor 8
        k_typical = 2.0e7  # 20 MN/m
        self.story_stiffness = k_typical * np.ones(self.n_floors)
        self.story_stiffness[7] = 0.60 * k_typical  # 60% stiffness at floor 8

        self.damping_ratio = 0.015  # 1.5% damping

        # Build matrices
        self.M = self._build_mass_matrix()
        self.K = self._build_stiffness_matrix()
        self.C = self._build_damping_matrix()

        # Tune TMD to soft-story mode
        self._tune_tmd_to_soft_story()

        # Newmark parameters
        self.beta = 0.25
        self.gamma = 0.5

        # Observation space: [floor8_disp, floor8_vel, tmd_disp, tmd_vel, roof_disp, roof_vel]
        # Focus on floor 8 (soft story) as primary observation
        if obs_bounds is None:
            obs_bounds = {
                'disp': 5.0, 'vel': 20.0, 'tmd_disp': 15.0, 'tmd_vel': 60.0
            }

        self.observation_space = spaces.Box(
            low=np.array([
                -obs_bounds['disp'], -obs_bounds['vel'],      # Floor 8
                -obs_bounds['tmd_disp'], -obs_bounds['tmd_vel'],  # TMD
                -obs_bounds['disp'], -obs_bounds['vel']       # Roof
            ]),
            high=np.array([
                obs_bounds['disp'], obs_bounds['vel'],
                obs_bounds['tmd_disp'], obs_bounds['tmd_vel'],
                obs_bounds['disp'], obs_bounds['vel']
            ]),
            dtype=np.float32
        )

        # Action space: TMD control force (normalized -1 to 1)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )

        # State tracking
        self.current_step = 0
        self.max_steps = len(earthquake_data)

        # Episode metrics tracking
        self.peak_displacements = None
        self.drift_history = None
        self.force_history = None

    def _tune_tmd_to_soft_story(self):
        """
        Tune TMD to the mode that most affects floor 8 (soft story)

        This targets the local soft-story deformation mode rather than
        the global first mode.
        """
        # Eigenvalue analysis of building (without TMD)
        M_bldg = self.M[:self.n_floors, :self.n_floors]
        K_bldg = self.K[:self.n_floors, :self.n_floors]

        eigenvalues, eigenvectors = np.linalg.eig(np.linalg.solve(M_bldg, K_bldg))
        omega_squared = np.real(eigenvalues)

        # Sort by frequency
        idx = np.argsort(omega_squared)
        omega_sorted = np.sqrt(omega_squared[idx])
        modes_sorted = eigenvectors[:, idx]

        # Find mode with largest participation at floor 8
        floor8_participation = np.abs(modes_sorted[self.tmd_floor, :])
        target_mode_idx = np.argmax(floor8_participation[:4])  # Check first 4 modes

        omega_target = omega_sorted[target_mode_idx]

        # Den Hartog optimal tuning
        mu = self.tmd_mass / (self.n_floors * self.floor_mass)  # 0.33%
        f_ratio = 1 / (1 + mu)
        omega_tmd = f_ratio * omega_target

        # Optimal damping ratio
        zeta_opt = np.sqrt(3 * mu / (8 * (1 + mu)))

        # Calculate k and c
        self.tmd_k = self.tmd_mass * omega_tmd**2
        self.tmd_c = 2 * zeta_opt * np.sqrt(self.tmd_k * self.tmd_mass)

        # Rebuild matrices with tuned TMD
        self.K = self._build_stiffness_matrix()
        self.C = self._build_damping_matrix()

    def _build_mass_matrix(self) -> np.ndarray:
        """Build mass matrix with TMD at floor 8"""
        M = np.zeros((13, 13))

        # Building masses
        for i in range(self.n_floors):
            M[i, i] = self.floor_mass

        # TMD mass
        M[12, 12] = self.tmd_mass

        return M

    def _build_stiffness_matrix(self) -> np.ndarray:
        """Build stiffness matrix with TMD connected to floor 8"""
        K = np.zeros((13, 13))

        # Building stiffness (same as before)
        for i in range(self.n_floors):
            K[i, i] += self.story_stiffness[i]
            if i < self.n_floors - 1:
                K[i, i] += self.story_stiffness[i + 1]
                K[i, i + 1] = -self.story_stiffness[i + 1]
                K[i + 1, i] = -self.story_stiffness[i + 1]

        # TMD stiffness - CONNECTED TO FLOOR 8 (index 7)
        K[self.tmd_floor, self.tmd_floor] += self.tmd_k
        K[self.tmd_floor, 12] = -self.tmd_k
        K[12, self.tmd_floor] = -self.tmd_k
        K[12, 12] = self.tmd_k

        return K

    def _build_damping_matrix(self) -> np.ndarray:
        """Build damping matrix with TMD connected to floor 8"""
        # Rayleigh damping for building
        eigenvalues = np.linalg.eigvals(np.linalg.solve(self.M, self.K))
        omega = np.sqrt(np.real(eigenvalues[eigenvalues > 1e-10]))
        omega = np.sort(omega)

        if len(omega) < 2:
            raise ValueError(f"System has fewer than 2 positive eigenvalues")

        omega1 = omega[0]
        omega2 = omega[1]
        zeta = self.damping_ratio

        A = np.array([[1/(2*omega1), omega1/2],
                      [1/(2*omega2), omega2/2]])
        coeffs = np.linalg.solve(A, [zeta, zeta])
        alpha, beta = coeffs

        C = alpha * self.M + beta * self.K

        # TMD damping - CONNECTED TO FLOOR 8
        C[self.tmd_floor, self.tmd_floor] += self.tmd_c
        C[self.tmd_floor, 12] -= self.tmd_c
        C[12, self.tmd_floor] -= self.tmd_c
        C[12, 12] += self.tmd_c

        return C

    def _newmark_step(
        self,
        d: np.ndarray,
        v: np.ndarray,
        a_prev: np.ndarray,
        ground_accel: float,
        control_force: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Newmark time integration step"""
        # Force vector
        F = np.zeros(13)

        # Earthquake forcing (ground acceleration)
        for i in range(self.n_floors):
            F[i] = -self.floor_mass * ground_accel
        F[12] = -self.tmd_mass * ground_accel  # TMD also feels ground motion

        # Control force applied between TMD and floor 8
        F[self.tmd_floor] -= control_force
        F[12] += control_force

        # Newmark prediction
        d_pred = d + self.dt * v + (0.5 - self.beta) * self.dt**2 * a_prev
        v_pred = v + (1 - self.gamma) * self.dt * a_prev

        # Effective stiffness matrix
        K_eff = self.M + self.gamma * self.dt * self.C + self.beta * self.dt**2 * self.K

        # Effective force
        F_eff = F - self.C @ v_pred - self.K @ d_pred

        # Solve for acceleration
        a = np.linalg.solve(K_eff, F_eff)

        # Newmark correction
        d_new = d_pred + self.beta * self.dt**2 * a
        v_new = v_pred + self.gamma * self.dt * a

        return d_new, v_new, a

    def reset(self, seed=None, options=None):
        """Reset environment"""
        super().reset(seed=seed)

        self.current_step = 0

        # Initial conditions
        self.d = np.zeros(13)
        self.v = np.zeros(13)
        self.a = np.zeros(13)

        # Metrics tracking
        self.peak_displacements = np.zeros(self.n_floors)
        self.drift_history = []
        self.force_history = []

        # Get first observation
        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def _get_observation(self) -> np.ndarray:
        """
        Get observation: [floor8_disp, floor8_vel, tmd_disp, tmd_vel, roof_disp, roof_vel]
        """
        # Normalize observations
        floor8_disp_norm = self.d[self.tmd_floor] / 5.0
        floor8_vel_norm = self.v[self.tmd_floor] / 20.0
        tmd_disp_norm = self.d[12] / 15.0
        tmd_vel_norm = self.v[12] / 60.0
        roof_disp_norm = self.d[11] / 5.0
        roof_vel_norm = self.v[11] / 20.0

        obs = np.array([
            floor8_disp_norm,
            floor8_vel_norm,
            tmd_disp_norm,
            tmd_vel_norm,
            roof_disp_norm,
            roof_vel_norm
        ], dtype=np.float32)

        return np.clip(obs, -1.0, 1.0)

    def _get_info(self) -> dict:
        """Get info dict"""
        # Calculate current drift at floor 8
        floor8_drift = abs(self.d[self.tmd_floor] - (self.d[self.tmd_floor - 1] if self.tmd_floor > 0 else 0))

        return {
            'floor8_displacement': self.d[self.tmd_floor],
            'floor8_drift': floor8_drift,
            'roof_displacement': self.d[11],
            'tmd_displacement': self.d[12],
            'control_force': 0.0,  # Will be updated in step
            'step': self.current_step
        }

    def step(self, action):
        """Execute one step"""
        # Denormalize action to control force
        control_force = float(action[0]) * self.max_force

        # Get ground acceleration
        ground_accel = self.earthquake_data[self.current_step]

        # Newmark integration
        self.d, self.v, self.a = self._newmark_step(
            self.d, self.v, self.a, ground_accel, control_force
        )

        # Update metrics
        for i in range(self.n_floors):
            self.peak_displacements[i] = max(self.peak_displacements[i], abs(self.d[i]))

        # Calculate drift at floor 8
        floor8_drift = abs(self.d[self.tmd_floor] - (self.d[self.tmd_floor - 1] if self.tmd_floor > 0 else 0))
        self.drift_history.append(floor8_drift)
        self.force_history.append(abs(control_force))

        # Calculate reward
        reward = self._calculate_reward(control_force, floor8_drift)

        # Update step
        self.current_step += 1
        done = self.current_step >= self.max_steps
        truncated = False

        # Get next observation
        obs = self._get_observation()
        info = self._get_info()
        info['control_force'] = control_force

        return obs, reward, done, truncated, info

    def _calculate_reward(self, control_force: float, floor8_drift: float) -> float:
        """
        New reward function with specified target penalties

        Penalties (per step):
        - P_disp(t) = -(d_roof(t) / 0.14)²  [14cm target]
        - P_dcr(t) = -(DCR(t) / 1.15)²      [1.15 target]
        - P_isdr(t) = -(ISDR(t) / 0.004)²   [0.4% target]
        - P_force(t) = -(w_f(t) / w_fmax)²  [force utilization]

        With:
        - w_disp = 4.0
        - w_DCR = 4.0
        - w_ISDR = 1.5
        - w_force = 0.2
        """
        # Current displacements
        roof_disp = abs(self.d[11])  # meters
        floor8_disp = abs(self.d[self.tmd_floor])

        # Displacement penalty (roof)
        d_roof_target = 0.14  # 14 cm in meters
        P_disp = -((roof_disp / d_roof_target) ** 2)
        w_disp = 4.0

        # ISDR penalty (floor 8)
        story_height = 3.0  # 3m typical
        ISDR = floor8_drift / story_height
        ISDR_target = 0.004  # 0.4%
        P_isdr = -((ISDR / ISDR_target) ** 2)
        w_ISDR = 1.5

        # DCR penalty (estimate from current drifts)
        # Simple approximation: DCR ≈ max_drift / mean_drift
        if len(self.drift_history) > 10:
            recent_drifts = self.drift_history[-100:]
            max_drift = max(recent_drifts)
            sorted_drifts = sorted(recent_drifts)
            percentile_75 = sorted_drifts[int(0.75 * len(sorted_drifts))] if len(sorted_drifts) > 0 else 1e-6
            DCR = max_drift / max(percentile_75, 1e-6)
        else:
            DCR = 1.0

        DCR_target = 1.15
        P_dcr = -((DCR / DCR_target) ** 2)
        w_DCR = 4.0

        # Force utilization penalty
        force_utilization = abs(control_force) / self.max_force
        P_force = -(force_utilization ** 2)
        w_force = 0.2

        # Total reward
        r_t = (w_disp * P_disp +
               w_DCR * P_dcr +
               w_ISDR * P_isdr +
               w_force * P_force)

        return r_t * self.reward_scale

    def get_episode_metrics(self) -> dict:
        """Calculate final episode metrics"""
        # Max ISDR
        max_isdr = 0.0
        story_height = 3.0  # 3m

        for drift in self.drift_history:
            isdr = (drift / story_height) * 100  # Convert to percentage
            max_isdr = max(max_isdr, isdr)

        # DCR
        if len(self.drift_history) > 0:
            sorted_drifts = sorted(self.drift_history)
            max_drift = sorted_drifts[-1]
            percentile_75 = sorted_drifts[int(0.75 * len(sorted_drifts))]
            DCR = max_drift / max(percentile_75, 1e-6)
        else:
            DCR = 1.0

        # Mean force
        mean_force = np.mean(self.force_history) if len(self.force_history) > 0 else 0.0

        return {
            'max_isdr_percent': max_isdr,
            'DCR': DCR,
            'mean_force': mean_force,
            'rms_roof_displacement': np.sqrt(np.mean(self.d[11]**2)),
            'max_drift': max(self.drift_history) if len(self.drift_history) > 0 else 0.0
        }


def make_soft_story_tmd_env(
    earthquake_file: str,
    max_force: float = 300000.0,
    reward_scale: float = 1.0
) -> SoftStoryTMDEnv:
    """
    Factory function to create soft-story TMD environment

    Args:
        earthquake_file: Path to earthquake CSV file
        max_force: Maximum control force (default 300 kN)
        reward_scale: Fixed reward scale (default 1.0)

    Returns:
        SoftStoryTMDEnv instance
    """
    # Load earthquake data
    data = np.loadtxt(earthquake_file, delimiter=',', skiprows=1)

    if data.shape[1] >= 2:
        times = data[:, 0]
        accelerations = data[:, 1]
        dt = float(np.mean(np.diff(times)))
    else:
        accelerations = data.flatten()
        dt = 0.02

    # Extract earthquake name
    import os
    eq_name = os.path.basename(earthquake_file).replace('.csv', '')

    return SoftStoryTMDEnv(
        earthquake_data=accelerations,
        dt=dt,
        max_force=max_force,
        earthquake_name=eq_name,
        reward_scale=reward_scale
    )

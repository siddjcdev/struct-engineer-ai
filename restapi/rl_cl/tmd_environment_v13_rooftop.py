"""
TMD ENVIRONMENT v13 - ROOFTOP TMD WITH ALL-FLOOR DRIFT TRACKING
================================================================

v13 CRITICAL FIXES:
-------------------
1. TMD mounted AT ROOFTOP (floor 12, 0-indexed as 11)
   - Tuned to first mode (global mode) for whole-building control

2. Track ALL floor drifts (12 floors) in reward function and metrics
   - Calculate ISDR for all 12 floors every timestep
   - Penalize MAX ISDR across all floors (not just one floor)
   - Store per-floor drift history for accurate metrics

3. Proper DCR calculation:
   - DCR = max(all_floor_max_drifts) / mean(all_floor_max_drifts)
   - Calculated from actual per-floor max drifts over episode
   - Not estimated from percentiles

4. Updated reward function:
   - max_isdr_current = max ISDR across all 12 floors at current timestep
   - Weights: w_disp=3.0, w_DCR=3.0, w_ISDR=5.0 (safety critical), w_force=0.3
   - Targets: roof_disp=14cm, DCR=1.15, ISDR=0.4%, force utilization

5. Enhanced get_episode_metrics():
   - Returns per-floor ISDR values
   - Identifies critical floor (floor with max ISDR)
   - Proper DCR from actual floor drift data

6. Observation space:
   - Focused on roof (primary control target) + TMD state + floor 8 monitoring
   - [roof_disp, roof_vel, tmd_disp, tmd_vel, floor8_disp, floor8_vel]

7. Per-floor drift history tracking:
   - self.drift_history_per_floor[floor_idx] = list of drifts for that floor
   - Updated every timestep for all 12 floors

8. From v12:
   - 300 kN max force for aggressive control
   - Fixed reward_scale=1.0
   - Same network architecture compatibility
   - Newmark integration with beta=0.25, gamma=0.5

RATIONALE:
----------
The v12 soft-story placement didn't properly track drift distribution across
the building. This v13 version:
- Places TMD at roof for global mode control (standard practice)
- Monitors and penalizes ALL floor drifts, not just one
- Properly calculates DCR as a measure of drift concentration
- Provides comprehensive per-floor metrics for structural safety assessment

Author: Claude Code
Date: January 2026
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple


class RooftopTMDEnv(gym.Env):
    """
    TMD Environment with TMD at rooftop and comprehensive all-floor drift tracking

    Key features:
    - TMD mounted at roof (floor 12) for global mode control
    - All 12 floor drifts calculated and tracked every timestep
    - Reward function penalizes max ISDR across all floors
    - Proper DCR calculation from per-floor drift data
    - Enhanced episode metrics with per-floor analysis
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
        self.story_height = 3.0  # 3m typical story height

        # TMD configuration - MOUNTED AT ROOFTOP (FLOOR 12)
        self.tmd_floor = 11  # Floor 12 (0-indexed as 11)
        self.tmd_mass = 0.04 * self.floor_mass  # 4% of floor mass = 8000 kg

        # TMD tuning - will be tuned to first mode (global mode)
        # Placeholder values - will be updated in _tune_tmd_to_first_mode()
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

        # Tune TMD to first mode (global mode)
        self._tune_tmd_to_first_mode()

        # Newmark parameters
        self.beta = 0.25
        self.gamma = 0.5

        # Observation space: [roof_disp, roof_vel, tmd_disp, tmd_vel, floor8_disp, floor8_vel]
        # Roof first (primary control target), TMD state, floor 8 monitoring (soft story)
        if obs_bounds is None:
            obs_bounds = {
                'disp': 5.0, 'vel': 20.0, 'tmd_disp': 15.0, 'tmd_vel': 60.0
            }

        # EXPANDED OBSERVATION SPACE for ISDR/DCR optimization
        # Include multiple floors so agent can observe drift patterns
        # Floors monitored: 1 (bottom), 4 (lower-mid), 8 (soft-story), 11 (top), 12 (roof)
        self.observation_space = spaces.Box(
            low=np.array([
                -obs_bounds['disp'], -obs_bounds['vel'],      # Floor 1 (bottom)
                -obs_bounds['disp'], -obs_bounds['vel'],      # Floor 4 (lower-mid)
                -obs_bounds['disp'], -obs_bounds['vel'],      # Floor 8 (soft-story - critical)
                -obs_bounds['disp'], -obs_bounds['vel'],      # Floor 11 (near-top)
                -obs_bounds['disp'], -obs_bounds['vel'],      # Floor 12 (roof)
                -obs_bounds['tmd_disp'], -obs_bounds['tmd_vel'],  # TMD state
            ]),
            high=np.array([
                obs_bounds['disp'], obs_bounds['vel'],
                obs_bounds['disp'], obs_bounds['vel'],
                obs_bounds['disp'], obs_bounds['vel'],
                obs_bounds['disp'], obs_bounds['vel'],
                obs_bounds['disp'], obs_bounds['vel'],
                obs_bounds['tmd_disp'], obs_bounds['tmd_vel'],
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
        self.drift_history_per_floor = None  # NEW: List of lists for per-floor drift tracking
        self.force_history = None

    def _tune_tmd_to_first_mode(self):
        """
        Tune TMD to the first mode (global mode) of the building

        Since TMD is at rooftop, we tune to the fundamental mode which
        has maximum participation at the roof. This provides global
        building control rather than targeting a specific floor.
        """
        # Eigenvalue analysis of building (without TMD)
        M_bldg = self.M[:self.n_floors, :self.n_floors]
        K_bldg = self.K[:self.n_floors, :self.n_floors]

        eigenvalues, eigenvectors = np.linalg.eig(np.linalg.solve(M_bldg, K_bldg))
        omega_squared = np.real(eigenvalues)

        # Sort by frequency
        idx = np.argsort(omega_squared)
        omega_sorted = np.sqrt(omega_squared[idx])

        # First mode (lowest frequency)
        omega_target = omega_sorted[0]

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
        """Build mass matrix with TMD at rooftop (floor 12)"""
        M = np.zeros((13, 13))

        # Building masses
        for i in range(self.n_floors):
            M[i, i] = self.floor_mass

        # TMD mass
        M[12, 12] = self.tmd_mass

        return M

    def _build_stiffness_matrix(self) -> np.ndarray:
        """Build stiffness matrix with TMD connected to rooftop (floor 12)"""
        K = np.zeros((13, 13))

        # Building stiffness (tridiagonal structure)
        for i in range(self.n_floors):
            K[i, i] += self.story_stiffness[i]
            if i < self.n_floors - 1:
                K[i, i] += self.story_stiffness[i + 1]
                K[i, i + 1] = -self.story_stiffness[i + 1]
                K[i + 1, i] = -self.story_stiffness[i + 1]

        # TMD stiffness - CONNECTED TO ROOFTOP (index 11)
        K[self.tmd_floor, self.tmd_floor] += self.tmd_k
        K[self.tmd_floor, 12] = -self.tmd_k
        K[12, self.tmd_floor] = -self.tmd_k
        K[12, 12] = self.tmd_k

        return K

    def _build_damping_matrix(self) -> np.ndarray:
        """Build damping matrix with TMD connected to rooftop"""
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

        # TMD damping - CONNECTED TO ROOFTOP
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

        # Control force applied between TMD and rooftop
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

        # NEW: Per-floor drift history - list of lists
        # drift_history_per_floor[floor_idx] = [drift1, drift2, ...]
        self.drift_history_per_floor = [[] for _ in range(self.n_floors)]

        self.force_history = []

        # Get first observation
        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def _get_observation(self) -> np.ndarray:
        """
        EXPANDED OBSERVATION for ISDR/DCR optimization.

        Returns: [floor1_disp, floor1_vel, floor4_disp, floor4_vel,
                  floor8_disp, floor8_vel, floor11_disp, floor11_vel,
                  roof_disp, roof_vel, tmd_disp, tmd_vel]

        Observing multiple floors allows the agent to:
        - Detect drift patterns across the building height
        - Identify which floors have high ISDR (soft-story floor 8 especially)
        - Optimize DCR by seeing drift distribution
        - Apply control to minimize max ISDR across all floors
        """
        # Normalize observations from 5 key floors + TMD
        floor1_disp_norm = self.d[0] / 5.0    # Bottom floor
        floor1_vel_norm = self.v[0] / 20.0
        floor4_disp_norm = self.d[3] / 5.0    # Lower-mid floor
        floor4_vel_norm = self.v[3] / 20.0
        floor8_disp_norm = self.d[7] / 5.0    # Soft-story (CRITICAL)
        floor8_vel_norm = self.v[7] / 20.0
        floor11_disp_norm = self.d[10] / 5.0  # Near-top floor
        floor11_vel_norm = self.v[10] / 20.0
        roof_disp_norm = self.d[11] / 5.0     # Roof (TMD attachment point)
        roof_vel_norm = self.v[11] / 20.0
        tmd_disp_norm = self.d[12] / 15.0     # TMD state
        tmd_vel_norm = self.v[12] / 60.0

        obs = np.array([
            floor1_disp_norm,
            floor1_vel_norm,
            floor4_disp_norm,
            floor4_vel_norm,
            floor8_disp_norm,
            floor8_vel_norm,
            floor11_disp_norm,
            floor11_vel_norm,
            roof_disp_norm,
            roof_vel_norm,
            tmd_disp_norm,
            tmd_vel_norm
        ], dtype=np.float32)

        return np.clip(obs, -1.0, 1.0)

    def _get_info(self) -> dict:
        """Get info dict"""
        # Calculate current drift at floor 8 (for info only)
        floor8_drift = abs(self.d[7] - self.d[6])

        return {
            'roof_displacement': self.d[11],
            'floor8_displacement': self.d[7],
            'floor8_drift': floor8_drift,
            'tmd_displacement': self.d[12],
            'control_force': 0.0,  # Will be updated in step
            'step': self.current_step
        }

    def _calculate_all_floor_drifts(self) -> np.ndarray:
        """
        Calculate drift at all 12 floors at current timestep

        Returns:
            drifts: np.ndarray of shape (12,) with drift at each floor
        """
        drifts = np.zeros(self.n_floors)

        for floor in range(self.n_floors):
            if floor == 0:
                # First floor drift relative to ground
                drifts[floor] = abs(self.d[floor])
            else:
                # Inter-story drift
                drifts[floor] = abs(self.d[floor] - self.d[floor - 1])

        return drifts

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

        # Update peak displacements
        for i in range(self.n_floors):
            self.peak_displacements[i] = max(self.peak_displacements[i], abs(self.d[i]))

        # NEW: Calculate and store drift for ALL floors at this timestep
        all_floor_drifts = self._calculate_all_floor_drifts()
        for floor in range(self.n_floors):
            self.drift_history_per_floor[floor].append(all_floor_drifts[floor])

        self.force_history.append(abs(control_force))

        # Calculate reward using all floor drifts
        reward = self._calculate_reward(control_force, all_floor_drifts)

        # Update step
        self.current_step += 1
        done = self.current_step >= self.max_steps
        truncated = False

        # Get next observation
        obs = self._get_observation()
        info = self._get_info()
        info['control_force'] = control_force

        return obs, reward, done, truncated, info

    def _calculate_reward(self, control_force: float, all_floor_drifts: np.ndarray) -> float:
        """
        AGGRESSIVE OPTIMIZATION STRATEGY:
        "Aggressively minimize soft-story ISDR and DCR, while keeping displacement
        reasonably below 17 cm and control forces within limits."

        Strategy:
        1. ISDR and DCR are PRIMARY - optimize these aggressively
        2. Displacement is a SOFT CONSTRAINT - keep reasonably below 17cm (aim for <15cm)
        3. Force is LIMITED - stay within actuator capacity (300kN)

        Reward Structure (aggressive optimization):
        - R_ISDR(t) = 1 - (max_ISDR(t) / 0.0055)²  [AGGRESSIVE: target 0.55% ISDR]
        - R_dcr(t) = 1 - (|DCR(t) - 1.0| / 0.5)²   [AGGRESSIVE: target 1.0 DCR, steep penalty]
        - R_disp(t) = 1 - (d_roof(t) / 0.15)²      [MODERATE: keep below 15cm comfortably]
        - P_disp_soft = penalty if > 15cm          [Soft constraint: discourage but allow]
        - P_disp_hard = HUGE penalty if > 17cm     [Hard constraint: never exceed]
        - P_force(t) = -(w_f(t) / w_fmax)²         [Keep forces reasonable]

        Weights (aggressive ISDR/DCR optimization):
        - w_ISDR = 15.0   # DOMINANT - aggressive ISDR minimization
        - w_DCR = 12.0    # VERY HIGH - aggressive DCR optimization
        - w_disp = 4.0    # MODERATE - keep displacement reasonable
        - w_disp_soft = 10.0   # Discourage 15-17cm range
        - w_disp_hard = 200.0  # Never exceed 17cm
        - w_force = 0.5   # Mild efficiency incentive

        Combined ratio: ISDR+DCR = 27.0 vs displacement = 4.0 → 6.75:1 aggressive optimization

        Target Performance:
        - ISDR: < 0.55% (aggressive minimization)
        - DCR: ~1.0 (aggressive uniformity optimization)
        - Displacement: < 15 cm comfortable, < 17 cm absolute limit
        - Force: < 300 kN (within actuator limits)

        Args:
            control_force: Current control force
            all_floor_drifts: Array of drifts at all 12 floors (current timestep)
        """
        # Current roof displacement
        roof_disp = abs(self.d[11])  # meters

        # HARD CONSTRAINT: Never exceed 17 cm (absolute limit)
        d_hard_limit = 0.17  # 17 cm - ABSOLUTE MAXIMUM
        if roof_disp > d_hard_limit:
            violation_ratio = (roof_disp - d_hard_limit) / d_hard_limit
            P_disp_hard = -200.0 * (violation_ratio ** 2)  # Catastrophic penalty
        else:
            P_disp_hard = 0.0

        # SOFT CONSTRAINT: Discourage 15-17cm range (prefer staying below 15cm)
        d_soft_target = 0.15  # 15 cm - comfortable target
        if roof_disp > d_soft_target and roof_disp <= d_hard_limit:
            # Linearly increasing penalty from 15-17cm
            soft_violation = (roof_disp - d_soft_target) / (d_hard_limit - d_soft_target)
            P_disp_soft = -10.0 * (soft_violation ** 2)
        else:
            P_disp_soft = 0.0

        # Displacement reward - reward for staying comfortably below 15cm
        disp_ratio = min(roof_disp / d_soft_target, 2.0)
        R_disp = 1.0 - (disp_ratio ** 2)
        w_disp = 4.0  # Moderate weight

        # ISDR reward - AGGRESSIVE PRIMARY OBJECTIVE
        # Maximize reward for low ISDR, steep penalty for high ISDR
        current_isdrs = all_floor_drifts / self.story_height
        max_isdr_current = np.max(current_isdrs)

        ISDR_target = 0.0055  # 0.55% target
        isdr_ratio = min(max_isdr_current / ISDR_target, 3.0)
        R_isdr = 1.0 - (isdr_ratio ** 2)
        w_ISDR = 15.0  # DOMINANT weight - aggressive optimization

        # DCR reward - AGGRESSIVE PRIMARY OBJECTIVE
        # CORRECTED: DCR = max(Δᵢ) / mean(Δᵢ) at current timestep
        # This is the proper structural engineering definition from ASCE standards

        # Use current floor drifts (already calculated)
        abs_drifts = np.abs(all_floor_drifts)
        max_current_drift = np.max(abs_drifts)
        mean_current_drift = np.mean(abs_drifts)

        # Calculate DCR using proper formula
        if mean_current_drift > 1e-8:  # Avoid division by zero
            DCR = max_current_drift / mean_current_drift
        else:
            DCR = 1.0  # At zero drift, assume uniform (perfect DCR)

        # Aggressive DCR optimization: steep quadratic penalty for deviation from 1.0
        DCR_ideal = 1.0
        # Steeper penalty: normalize deviation by 0.5 instead of 1.15
        dcr_deviation = min(abs(DCR - DCR_ideal) / 0.5, 1.5)
        R_dcr = 1.0 - (dcr_deviation ** 2)
        w_DCR = 12.0  # VERY HIGH weight - aggressive optimization

        # Force utilization penalty (keep forces reasonable)
        force_utilization = abs(control_force) / self.max_force
        P_force = -(force_utilization ** 2)
        w_force = 0.5

        # Total reward - AGGRESSIVE optimization of ISDR and DCR
        # Priority hierarchy:
        # 1. Never exceed 17cm displacement (-200× penalty if violated)
        # 2. AGGRESSIVELY minimize ISDR (15.0 weight)
        # 3. AGGRESSIVELY minimize DCR deviation from 1.0 (12.0 weight)
        # 4. Stay comfortably below 15cm displacement (4.0 weight + 10.0 soft penalty)
        # 5. Keep forces reasonable (0.5 weight)
        r_t = (w_ISDR * R_isdr +           # 15.0 * [0 to 1] = 0 to 15
               w_DCR * R_dcr +              # 12.0 * [0 to 1] = 0 to 12
               w_disp * R_disp +            # 4.0 * [0 to 1] = 0 to 4
               P_disp_soft +                # -10.0 * (0 to 1) if 15-17cm
               P_disp_hard +                # -200.0 * violation² if >17cm
               w_force * P_force)           # 0.5 * [-1 to 0] = -0.5 to 0

        return r_t * self.reward_scale

    def get_episode_metrics(self) -> dict:
        """
        Calculate final episode metrics with comprehensive per-floor analysis

        NEW in v13:
        - Returns per-floor ISDR values for all 12 floors
        - Identifies critical floor (floor with maximum ISDR)
        - Proper DCR calculation: max(floor_max_drifts) / mean(floor_max_drifts)
        - Floor-by-floor drift analysis

        Returns:
            dict with keys:
                - max_isdr_percent: Maximum ISDR across all floors (%)
                - critical_floor: Floor number (1-12) with maximum ISDR
                - floor_isdrs: List of max ISDR for each floor (%)
                - DCR: Proper drift concentration ratio
                - max_roof_displacement_cm: Maximum roof displacement (cm)
                - mean_force: Mean control force (N)
                - max_force: Maximum control force (N)
                - rms_roof_displacement: RMS roof displacement (m)
                - max_drift_per_floor: List of max drift at each floor (m)
        """
        floor_isdrs = []
        floor_max_drifts = []

        # Calculate max ISDR for each floor over the episode
        for floor in range(self.n_floors):
            floor_drifts = self.drift_history_per_floor[floor]

            if len(floor_drifts) > 0:
                max_drift = max(floor_drifts)
            else:
                max_drift = 0.0

            # Convert to ISDR (percentage)
            max_isdr = (max_drift / self.story_height) * 100

            floor_isdrs.append(max_isdr)
            floor_max_drifts.append(max_drift)

        # Overall max ISDR and critical floor
        max_isdr_overall = max(floor_isdrs)
        critical_floor = floor_isdrs.index(max_isdr_overall) + 1  # 1-indexed for reporting

        # Proper DCR calculation
        # DCR = max drift across all floors / mean drift across all floors
        if len(floor_max_drifts) > 0 and max(floor_max_drifts) > 0:
            DCR = max(floor_max_drifts) / max(np.mean(floor_max_drifts), 1e-6)
        else:
            DCR = 1.0

        # Force statistics
        mean_force = np.mean(self.force_history) if len(self.force_history) > 0 else 0.0
        max_force = np.max(self.force_history) if len(self.force_history) > 0 else 0.0

        # Roof displacement statistics
        max_roof_displacement_cm = np.max(self.peak_displacements) * 100

        # RMS roof displacement (calculated from displacement history if available)
        # For now, use peak as approximation
        rms_roof_displacement = max_roof_displacement_cm / 100 / np.sqrt(2)

        return {
            'max_isdr_percent': max_isdr_overall,
            'critical_floor': critical_floor,
            'floor_isdrs': floor_isdrs,
            'DCR': DCR,
            'max_roof_displacement_cm': max_roof_displacement_cm,
            'mean_force': mean_force,
            'max_force': max_force,
            'rms_roof_displacement': rms_roof_displacement,
            'max_drift_per_floor': floor_max_drifts,
            'earthquake_name': self.earthquake_name
        }


def make_rooftop_tmd_env(
    earthquake_file: str,
    max_force: float = 300000.0,
    reward_scale: float = 1.0
) -> RooftopTMDEnv:
    """
    Factory function to create rooftop TMD environment with all-floor tracking

    Args:
        earthquake_file: Path to earthquake CSV file
        max_force: Maximum control force (default 300 kN)
        reward_scale: Fixed reward scale (default 1.0)

    Returns:
        RooftopTMDEnv instance
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

    return RooftopTMDEnv(
        earthquake_data=accelerations,
        dt=dt,
        max_force=max_force,
        earthquake_name=eq_name,
        reward_scale=reward_scale
    )

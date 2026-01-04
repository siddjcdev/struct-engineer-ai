"""
PERFECT RL API - COMPLETE SERVER
=================================

All-in-one API server for your champion Perfect RL model
Just run: python perfect_rl_api.py

Author: Siddharth
Date: December 2025
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import uvicorn
import numpy as np
import torch
from stable_baselines3 import SAC, PPO
import time
import os


# ================================================================
# MODEL WRAPPER
# ================================================================

class RLCLController:
    """RL CL model wrapper"""
    
    def __init__(self, model_path: str):
        print(f"RLCLController: Loading RL CL model from {model_path}...")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"RLCLController: Model not found: {model_path}")

        # Auto-detect model type (SAC or PPO)
        # Try loading as PPO first, fall back to SAC
        model_type = None
        try:
            self.model = PPO.load(model_path, device='cpu')
            model_type = "PPO"
        except Exception as e_ppo:
            try:
                self.model = SAC.load(model_path, device='cpu')
                model_type = "SAC"
            except Exception as e_sac:
                raise RuntimeError(
                    f"Failed to load model as PPO or SAC:\n"
                    f"  PPO error: {e_ppo}\n"
                    f"  SAC error: {e_sac}"
                )

        self.model_type = model_type
        self.max_force = 150000.0  # 150 kN

        # CRITICAL FIX: Use proper observation bounds for 8-value expanded state space
        # Old bounds (±1.2m disp, ±3m/s vel) were too small and caused extreme clipping
        # Training uses: ±5.0m disp, ±20.0m/s vel for floors (roof, floor8, floor6), ±15.0m/±60m/s for TMD
        # This matches the rl_cl_tmd_environment.py defaults with obs_bounds = {'disp': 5.0, 'vel': 20.0, 'tmd_disp': 15.0, 'tmd_vel': 60.0}
        self.obs_bounds_array = np.array([
            [-5.0, -20.0, -5.0, -20.0, -5.0, -20.0, -15.0, -60.0],  # 8-value lower bounds
            [5.0, 20.0, 5.0, 20.0, 5.0, 20.0, 15.0, 60.0]           # 8-value upper bounds
        ], dtype=np.float32)

        # Legacy 4-value bounds - ALSO FIXED for backward compatibility
        # Was (±0.5m, ±2m/s) causing catastrophic clipping on extreme earthquakes
        self.obs_bounds = {
            'roof_disp': (-5.0, 5.0),      # ±5.0 m (matches training)
            'roof_vel': (-20.0, 20.0),     # ±20.0 m/s (matches training)
            'tmd_disp': (-15.0, 15.0),     # ±15.0 m (matches training)
            'tmd_vel': (-60.0, 60.0)       # ±60.0 m/s (matches training)
        }
        self.clip_warnings = 0
        self._last_force = 0.0  # For force rate limiting under latency

        print(f"✅ RLCLController: RL CL model loaded successfully!")
        print(f"   RLCLController:     Model type: {model_type}")
        print(f"   RLCLController:     Device: cpu")
        print("   RLCLController: RL CL performance:")
        print("   RLCLController:     • M7.4 (0.75g PGA): Target <35 cm (85-91% reduction)")
        print("   RLCLController:     • M8.4 (0.90g PGA): Target <45 cm (87-92% reduction)")
        print("   RLCLController:     • Trained with proper train/test split + curriculum")
        print(f"   RLCLController:     • Observation space: 8 values (roof, floor8, floor6, TMD)")
        print(f"   RLCLController:     • Bounds: ±5.0m disp, ±20.0m/s vel (floors), ±15.0m TMD disp, ±60.0m/s TMD vel")
    
    def predict_single(self, roof_disp, roof_vel, tmd_disp, tmd_vel):
        """Single prediction with safety clipping AND rate limiting"""
        # SAFETY: Clip observations to training bounds
        roof_disp_clip = np.clip(roof_disp, *self.obs_bounds['roof_disp'])
        roof_vel_clip = np.clip(roof_vel, *self.obs_bounds['roof_vel'])
        tmd_disp_clip = np.clip(tmd_disp, *self.obs_bounds['tmd_disp'])
        tmd_vel_clip = np.clip(tmd_vel, *self.obs_bounds['tmd_vel'])

        if (roof_disp_clip != roof_disp or roof_vel_clip != roof_vel or
            tmd_disp_clip != tmd_disp or tmd_vel_clip != tmd_vel):
            self.clip_warnings += 1
            if self.clip_warnings <= 5:  # Only print first 5 warnings
                print(f"⚠️  RL-CL SAFETY: Observation out of bounds (extreme earthquake)")
                print(f"    Original: roof_d={roof_disp:.3f}m, roof_v={roof_vel:.3f}m/s")
                print(f"    Clipped:  roof_d={roof_disp_clip:.3f}m, roof_v={roof_vel_clip:.3f}m/s")

        obs = np.array([roof_disp_clip, roof_vel_clip, tmd_disp_clip, tmd_vel_clip], dtype=np.float32)
        action, _ = self.model.predict(obs, deterministic=True)
        force = float(action[0]) * self.max_force
        force = np.clip(force, -self.max_force, self.max_force)
        
        # CRITICAL: Apply force rate limiting for latency robustness
        max_rate = 50000.0  # Max change per timestep (N)
        delta = force - self._last_force
        if abs(delta) > max_rate:
            force = self._last_force + np.sign(delta) * max_rate
        
        self._last_force = force
        return force
    
    def predict_batch(self, roof_disp_list, roof_vel_list, tmd_disp_list, tmd_vel_list):
        """Batch prediction with safety clipping and rate limiting"""
        n = len(roof_disp_list)
        forces = np.zeros(n)

        for i in range(n):
            # SAFETY: Clip each observation to training bounds
            roof_disp_clip = np.clip(roof_disp_list[i], *self.obs_bounds['roof_disp'])
            roof_vel_clip = np.clip(roof_vel_list[i], *self.obs_bounds['roof_vel'])
            tmd_disp_clip = np.clip(tmd_disp_list[i], *self.obs_bounds['tmd_disp'])
            tmd_vel_clip = np.clip(tmd_vel_list[i], *self.obs_bounds['tmd_vel'])

            obs = np.array([
                roof_disp_clip,
                roof_vel_clip,
                tmd_disp_clip,
                tmd_vel_clip
            ], dtype=np.float32)

            action, _ = self.model.predict(obs, deterministic=True)
            force = float(action[0]) * self.max_force
            force = np.clip(force, -self.max_force, self.max_force)
            
            # CRITICAL: Force rate limiting for latency robustness
            max_rate = 50000.0
            delta = force - self._last_force
            if abs(delta) > max_rate:
                force = self._last_force + np.sign(delta) * max_rate
            
            self._last_force = force
            forces[i] = force

        return forces

    def simulate_episode(self, earthquake_data, dt=0.02):
        """
        Run full episode simulation with the controller

        Args:
            earthquake_data: Ground acceleration time series
            dt: Time step (default 0.02s)

        Returns:
            dict with forces and all performance metrics
        """
        try:
            # Import here to avoid circular dependency
            import sys
            import os
            # Add path to rl_cl module
            # rl_cl_path = os.path.join(os.path.dirname(__file__), '..', '..', 'rl', 'rl_cl')
            # if rl_cl_path not in sys.path:
            #     sys.path.insert(0, rl_cl_path)

            #from .rl_cl_tmd_environment import ImprovedTMDBuildingEnv
            #from .tmd_environment_shaped_reward import ImprovedTMDBuildingEnv
            from .tmd_environment_ppo_wrapper import make_ppo_friendly_env


            # Create environment with SAME obs_bounds as training
            # CRITICAL: Must match train_final_robust_rl_cl.py obs_bounds
            obs_bounds = {
                'disp': 5.0,      # ±5.0m (same as training)
                'vel': 20.0,      # ±20.0m/s
                'tmd_disp': 15.0, # ±15.0m
                'tmd_vel': 60.0   # ±60.0m/s
            }
            # env = ImprovedTMDBuildingEnv(
            #     earthquake_data=earthquake_data,
            #     dt=dt,
            #     max_force=self.max_force,
            #     earthquake_name="API Simulation",
            #     obs_bounds=obs_bounds  # ADDED: Match training bounds
            # )
            #from tmd_environment_ppo_wrapper import make_ppo_friendly_env
            env = make_ppo_friendly_env(
                earthquake_data, 
                max_force=self.max_force,
                normalize_obs=True,      # Normalize observations
                clip_reward=10.0         # Clip extreme rewards
            )

            # Run episode
            obs, info = env.reset()
            done = False
            forces = []

            while not done:
                # SAFETY: Clip observation to training bounds before inference
                # FIXED: Now handles 8-value observations correctly
                # obs = [roof_disp, roof_vel, floor8_disp, floor8_vel, floor6_disp, floor6_vel, tmd_disp, tmd_vel]
                obs_clipped = np.clip(obs, self.obs_bounds_array[0], self.obs_bounds_array[1])

                # Get action from model
                action, _ = self.model.predict(obs_clipped, deterministic=True)
                forces.append(float(action[0]) * self.max_force)

                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

            # Get metrics
            metrics = env.get_episode_metrics()

            # Add forces to metrics
            metrics['forces'] = forces
            metrics['forces_N'] = forces
            metrics['forces_kN'] = [f/1000 for f in forces]

            return metrics

        except Exception as e:
            print(f"Error in simulate_episode: {e}")
            import traceback
            traceback.print_exc()
            raise

# ============================================================================
# Reinforcement Learning Model Definition
# ============================================================================

# Request/Response models

class RLCLSingleRequest(BaseModel):
    """Single prediction request"""
    roof_displacement: float = Field(..., description="Roof displacement (m)")
    roof_velocity: float = Field(..., description="Roof velocity (m/s)")
    tmd_displacement: float = Field(..., description="TMD displacement (m)")
    tmd_velocity: float = Field(..., description="TMD velocity (m/s)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "roof_displacement": 0.15,
                "roof_velocity": 0.8,
                "tmd_displacement": 0.16,
                "tmd_velocity": 0.9
            }
        }


class RLCLSingleResponse(BaseModel):
    """Single prediction response"""
    force_N: float = Field(..., description="Control force (Newtons)")
    force_kN: float = Field(..., description="Control force (kilonewtons)")
    inference_time_ms: float = Field(..., description="Inference time (ms)")


class RLCLBatchRequest(BaseModel):
    """Batch prediction request"""
    roof_displacements: List[float] = Field(..., description="Roof displacements (m)")
    roof_velocities: List[float] = Field(..., description="Roof velocities (m/s)")
    tmd_displacements: List[float] = Field(..., description="TMD displacements (m)")
    tmd_velocities: List[float] = Field(..., description="TMD velocities (m/s)")

    class Config:
        json_schema_extra = {
            "example": {
                "roof_displacements": [0.15, -0.10, 0.05],
                "roof_velocities": [0.8, -0.5, 0.3],
                "tmd_displacements": [0.16, -0.09, 0.06],
                "tmd_velocities": [0.85, -0.48, 0.32]
            }
        }


class RLCLSimulationRequest(BaseModel):
    """Full simulation request"""
    earthquake_data: List[float] = Field(..., description="Ground acceleration time series (m/s²)")
    dt: float = Field(0.02, description="Time step (seconds)")

    class Config:
        json_schema_extra = {
            "example": {
                "earthquake_data": [0.5, 0.8, 1.2, 1.0, 0.6],
                "dt": 0.02
            }
        }


class RLCLBatchResponse(BaseModel):
    """Batch prediction response"""
    forces_N: List[float] = Field(..., description="Control forces (Newtons)")
    forces_kN: List[float] = Field(..., description="Control forces (kilonewtons)")
    count: int = Field(..., description="Number of predictions")
    total_time_ms: float = Field(..., description="Total time (ms)")
    avg_time_ms: float = Field(..., description="Average time per prediction (ms)")
    model: str = "Perfect RL (Champion)"

    # Optional metrics (populated when running full simulation)
    rms_roof_displacement: float = Field(None, description="RMS of roof displacement (m)")
    peak_roof_displacement: float = Field(None, description="Peak roof displacement (m)")
    max_drift: float = Field(None, description="Maximum interstory drift (m)")
    DCR: float = Field(None, description="Drift Concentration Ratio")
    peak_force_kN: float = Field(None, description="Peak control force (kN)")
    mean_force_kN: float = Field(None, description="Mean absolute control force (kN)")


class RLCLSimulationResponse(BaseModel):
    """Full simulation response with all metrics"""
    forces_N: List[float] = Field(..., description="Control forces (Newtons)")
    forces_kN: List[float] = Field(..., description="Control forces (kilonewtons)")
    count: int = Field(..., description="Number of timesteps")

    # Performance metrics
    rms_roof_displacement: float = Field(..., description="RMS of roof displacement (m)")
    peak_roof_displacement: float = Field(..., description="Peak roof displacement (m)")
    max_drift: float = Field(..., description="Maximum interstory drift (m)")
    DCR: float = Field(..., description="Drift Concentration Ratio")
    peak_force: float = Field(..., description="Peak control force (N)")
    mean_force: float = Field(..., description="Mean absolute control force (N)")
    peak_force_kN: float = Field(..., description="Peak control force (kN)")
    mean_force_kN: float = Field(..., description="Mean absolute control force (kN)")
    peak_disp_by_floor: List[float] = Field(default=[], description="Peak displacement at each floor (m)")
    rms_roof_accel: float = Field(default=0.0, description="RMS roof acceleration (m/s²)")

    # Metadata
    model: str = "Perfect RL (Champion)"
    simulation_time_ms: float = Field(..., description="Total simulation time (ms)")

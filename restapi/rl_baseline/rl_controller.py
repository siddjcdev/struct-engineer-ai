"""
RL CONTROLLER FOR API DEPLOYMENT
================================

Wrapper for trained RL model to use in FastAPI endpoint

Author: Siddharth
Date: December 2025
"""

import numpy as np
from pydantic import BaseModel, Field
import torch
from stable_baselines3 import SAC
from typing import List, Union


class RLTMDController:
    """
    Reinforcement Learning TMD Controller for API deployment
    
    Loads trained SAC model and provides simple interface for predictions
    """
    
    def __init__(self, model_path: str):
        """
        Initialize RL controller

        Args:
            model_path: Path to trained .zip model file
        """
        print(f"RLTMDController: Loading RL model from {model_path}...")

        # Load trained SAC model
        self.model = SAC.load(model_path, device='cpu')

        # Force limits
        self.max_force = 100000.0  # 100 kN in Newtons

        # SAFETY: Observation bounds (MUST match training environment)
        # These bounds prevent out-of-distribution inputs on extreme earthquakes
        self.obs_bounds = {
            'roof_disp': (-0.5, 0.5),      # ±50 cm
            'roof_vel': (-2.0, 2.0),       # ±2.0 m/s
            'tmd_disp': (-0.6, 0.6),       # ±60 cm
            'tmd_vel': (-2.5, 2.5)         # ±2.5 m/s
        }
        self.clip_warnings = 0  # Track how many times we clip observations

        print(f"✅ RLTMDController: RL model loaded successfully")
        print(f"   RLTMDController:     Model type: {type(self.model).__name__}")
        print(f"   RLTMDController:     Device: {self.model.device}")
        print(f"   RLTMDController:     Observation bounds: roof_disp={self.obs_bounds['roof_disp']}, "
              f"roof_vel={self.obs_bounds['roof_vel']}")
    
    
    def predict(
        self,
        roof_displacement: float,
        roof_velocity: float,
        tmd_displacement: float,
        tmd_velocity: float,
        deterministic: bool = True
    ) -> float:
        """
        Predict control force for given state

        Args:
            roof_displacement: Roof displacement (meters)
            roof_velocity: Roof velocity (m/s)
            tmd_displacement: TMD displacement (meters)
            tmd_velocity: TMD velocity (m/s)
            deterministic: Use deterministic policy (recommended for deployment)

        Returns:
            Control force in Newtons
        """

        # SAFETY: Clip observations to training bounds
        # This prevents catastrophic failures on extreme earthquakes
        clipped = False

        roof_disp_clip = np.clip(roof_displacement, *self.obs_bounds['roof_disp'])
        roof_vel_clip = np.clip(roof_velocity, *self.obs_bounds['roof_vel'])
        tmd_disp_clip = np.clip(tmd_displacement, *self.obs_bounds['tmd_disp'])
        tmd_vel_clip = np.clip(tmd_velocity, *self.obs_bounds['tmd_vel'])

        if (roof_disp_clip != roof_displacement or roof_vel_clip != roof_velocity or
            tmd_disp_clip != tmd_displacement or tmd_vel_clip != tmd_velocity):
            clipped = True
            self.clip_warnings += 1
            if self.clip_warnings <= 5:  # Only print first 5 warnings
                print(f"⚠️  RL SAFETY: Observation out of bounds, clipping to training range")
                print(f"    Original: roof_d={roof_displacement:.3f}, roof_v={roof_velocity:.3f}, "
                      f"tmd_d={tmd_displacement:.3f}, tmd_v={tmd_velocity:.3f}")
                print(f"    Clipped:  roof_d={roof_disp_clip:.3f}, roof_v={roof_vel_clip:.3f}, "
                      f"tmd_d={tmd_disp_clip:.3f}, tmd_v={tmd_vel_clip:.3f}")

        # Create observation (same format as training)
        obs = np.array([
            roof_disp_clip,
            roof_vel_clip,
            tmd_disp_clip,
            tmd_vel_clip
        ], dtype=np.float32)

        # Get action from policy (returns normalized force in [-1, 1])
        action, _ = self.model.predict(obs, deterministic=deterministic)

        # Scale to actual force
        force_N = float(action[0]) * self.max_force

        # Clamp to limits (safety)
        force_N = np.clip(force_N, -self.max_force, self.max_force)
        return force_N
    
    
    def predict_batch(
        self,
        roof_displacements: Union[List[float], np.ndarray],
        roof_velocities: Union[List[float], np.ndarray],
        tmd_displacements: Union[List[float], np.ndarray],
        tmd_velocities: Union[List[float], np.ndarray],
        deterministic: bool = True
    ) -> np.ndarray:
        """
        Predict control forces for batch of states
        
        Args:
            roof_displacements: Array of roof displacements (meters)
            roof_velocities: Array of roof velocities (m/s)
            tmd_displacements: Array of TMD displacements (meters)
            tmd_velocities: Array of TMD velocities (m/s)
            deterministic: Use deterministic policy
            
        Returns:
            Array of control forces in Newtons
        """
        
        # Convert to numpy arrays
        roof_disp = np.array(roof_displacements, dtype=np.float32)
        roof_vel = np.array(roof_velocities, dtype=np.float32)
        tmd_disp = np.array(tmd_displacements, dtype=np.float32)
        tmd_vel = np.array(tmd_velocities, dtype=np.float32)

        print(f"RL BATCH: Predicting batch of size {len(roof_disp)}")   
        
        # Compute forces for each timestep
        n = len(roof_disp)
        forces_N = np.zeros(n)
        
        for i in range(n):
            forces_N[i] = self.predict(
                roof_disp[i],
                roof_vel[i],
                tmd_disp[i],
                tmd_vel[i],
                deterministic=deterministic
            )
        
        print(f"RL BATCH: Completed predictions for batch of size {n}")
        return forces_N

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
            # Add path to rl_baseline module
            rl_baseline_path = os.path.join(os.path.dirname(__file__), '..', '..', 'rl', 'rl_baseline')
            if rl_baseline_path not in sys.path:
                sys.path.insert(0, rl_baseline_path)

            from tmd_environment import TMDBuildingEnv

            # Create environment
            env = TMDBuildingEnv(
                earthquake_data=earthquake_data,
                dt=dt,
                max_force=self.max_force,
                earthquake_name="API Simulation"
            )

            # Run episode
            obs, info = env.reset()
            done = False
            forces = []

            while not done:
                # SAFETY: Clip observation to training bounds before inference
                # This prevents catastrophic failures on extreme earthquakes
                obs_clipped = np.clip(obs,
                                     [self.obs_bounds['roof_disp'][0], self.obs_bounds['roof_vel'][0],
                                      self.obs_bounds['tmd_disp'][0], self.obs_bounds['tmd_vel'][0]],
                                     [self.obs_bounds['roof_disp'][1], self.obs_bounds['roof_vel'][1],
                                      self.obs_bounds['tmd_disp'][1], self.obs_bounds['tmd_vel'][1]])

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


# Request/Response models
class RLSingleRequest(BaseModel):
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


class RLSingleResponse(BaseModel):
    """Single prediction response"""
    force_N: float = Field(..., description="Control force (Newtons)")
    force_kN: float = Field(..., description="Control force (kilonewtons)")
    inference_time_ms: float = Field(..., description="Inference time (ms)")

class RLBatchRequest(BaseModel):
    roof_displacements: List[float]
    roof_velocities: List[float]
    tmd_displacements: List[float]
    tmd_velocities: List[float]


class RLBatchResponse(BaseModel):
    forces: List[float]  # Forces in kN
    force_unit: str = "kN"
    num_predictions: int
    inference_time_ms: float


class RLSimulationRequest(BaseModel):
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


class RLSimulationResponse(BaseModel):
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

    # Metadata
    model: str = "RL Baseline"
    simulation_time_ms: float = Field(..., description="Total simulation time (ms)")
    
# ================================================================
# TESTING
# ================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python rl_controller.py <path_to_model.zip>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    
    print("\n" + "="*60)
    print("  RL CONTROLLER TEST")
    print("="*60 + "\n")
    
    # Initialize controller
    controller = RLTMDController(model_path)
    
    # Test single prediction
    print("\n📋 Testing single prediction...")
    test_roof_disp = 0.15
    test_roof_vel = 0.8
    test_tmd_disp = 0.16
    test_tmd_vel = 0.9
    
    force_N = controller.predict(
        test_roof_disp,
        test_roof_vel,
        test_tmd_disp,
        test_tmd_vel
    )
    
    print(f"   Input state:")
    print(f"      Roof: disp={test_roof_disp}m, vel={test_roof_vel}m/s")
    print(f"      TMD:  disp={test_tmd_disp}m, vel={test_tmd_vel}m/s")
    print(f"   Output force: {force_N:.2f} N ({force_N/1000:.2f} kN)")
    
    # Test batch prediction
    print("\n📋 Testing batch prediction...")
    n_samples = 5
    roof_disp = np.random.uniform(-0.2, 0.2, n_samples)
    roof_vel = np.random.uniform(-1.0, 1.0, n_samples)
    tmd_disp = roof_disp + np.random.uniform(-0.05, 0.05, n_samples)
    tmd_vel = roof_vel + np.random.uniform(-0.1, 0.1, n_samples)
    
    forces_N = controller.predict_batch(roof_disp, roof_vel, tmd_disp, tmd_vel)
    forces_kN = forces_N / 1000
    
    print(f"   Batch size: {n_samples}")
    print(f"   Forces (kN): [{', '.join([f'{f:.2f}' for f in forces_kN])}]")
    print(f"   Force range: [{forces_kN.min():.2f}, {forces_kN.max():.2f}] kN")
    
    print("\n✅ Controller test complete!\n")

"""
RL CONTROLLER FOR API DEPLOYMENT
================================

Wrapper for trained RL model to use in FastAPI endpoint

Author: Siddharth
Date: December 2025
"""

import numpy as np
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
        print(f"Loading RL model from {model_path}...")
        
        # Load trained SAC model
        self.model = SAC.load(model_path, device='cpu')
        
        # Force limits
        self.max_force = 100000.0  # 100 kN in Newtons
        
        print(f"✅ RL model loaded successfully")
        print(f"   Model type: {type(self.model).__name__}")
        print(f"   Device: {self.model.device}")
    
    
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
        
        # Create observation (same format as training)
        obs = np.array([
            roof_displacement,
            roof_velocity,
            tmd_displacement,
            tmd_velocity
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
        
        return forces_N


# ================================================================
# FASTAPI ENDPOINT CODE
# ================================================================

"""
Add this to your main.py FastAPI application:

from rl_controller import RLTMDController

# At startup, initialize RL controller
rl_controller = RLTMDController("path/to/tmd_sac_final.zip")


# Request/Response models
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


# Endpoint
@app.post("/rl/predict-batch", response_model=RLBatchResponse)
async def rl_batch_predict(request: RLBatchRequest):
    import time
    start_time = time.time()
    
    try:
        # Validate input
        n = len(request.roof_displacements)
        if not (len(request.roof_velocities) == n and 
                len(request.tmd_displacements) == n and 
                len(request.tmd_velocities) == n):
            raise HTTPException(
                status_code=422,
                detail="All input arrays must have same length"
            )
        
        # Predict forces (returns Newtons)
        forces_N = rl_controller.predict_batch(
            request.roof_displacements,
            request.roof_velocities,
            request.tmd_displacements,
            request.tmd_velocities
        )
        
        # Convert to kN
        forces_kN = forces_N / 1000.0
        
        # Calculate inference time
        inference_time = (time.time() - start_time) * 1000  # ms
        
        return RLBatchResponse(
            forces=forces_kN.tolist(),
            force_unit="kN",
            num_predictions=len(forces_kN),
            inference_time_ms=inference_time
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"RL controller error: {str(e)}"
        )


@app.get("/rl/test")
async def test_rl_controller():
    # Test with sample state
    test_state = {
        "roof_disp": 0.15,
        "roof_vel": 0.8,
        "tmd_disp": 0.16,
        "tmd_vel": 0.9
    }
    
    force_N = rl_controller.predict(
        test_state["roof_disp"],
        test_state["roof_vel"],
        test_state["tmd_disp"],
        test_state["tmd_vel"]
    )
    
    return {
        "status": "ok",
        "test_input": test_state,
        "output": {
            "force_N": force_N,
            "force_kN": force_N / 1000.0
        }
    }
"""


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

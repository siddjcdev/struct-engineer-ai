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
from stable_baselines3 import SAC
import time
import os


# ================================================================
# MODEL WRAPPER
# ================================================================

class RLCLController:
    """Champion Perfect RL model wrapper"""
    
    def __init__(self, model_path: str):
        print(f"Loading Perfect RL model from {model_path}...")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load SAC model
        self.model = SAC.load(model_path, device='cpu')
        self.max_force = 150000.0  # 150 kN
        
        print("✅ Perfect RL model loaded!")
        print("   Champion performance:")
        print("   • TEST3 (M4.5): 24.67 cm (21.8% vs passive)")
        print("   • TEST4 (M6.9): 20.80 cm (32% vs passive)")
        print("   • Average: ~21.5 cm, Beats fuzzy by 14%")
    
    def predict_single(self, roof_disp, roof_vel, tmd_disp, tmd_vel):
        """Single prediction"""
        obs = np.array([roof_disp, roof_vel, tmd_disp, tmd_vel], dtype=np.float32)
        action, _ = self.model.predict(obs, deterministic=True)
        force = float(action[0]) * self.max_force
        return np.clip(force, -self.max_force, self.max_force)
    
    def predict_batch(self, roof_disp_list, roof_vel_list, tmd_disp_list, tmd_vel_list):
        """Batch prediction"""
        n = len(roof_disp_list)
        forces = np.zeros(n)
        
        for i in range(n):
            obs = np.array([
                roof_disp_list[i], 
                roof_vel_list[i],
                tmd_disp_list[i],
                tmd_vel_list[i]
            ], dtype=np.float32)
            
            action, _ = self.model.predict(obs, deterministic=True)
            forces[i] = float(action[0]) * self.max_force
        
        return np.clip(forces, -self.max_force, self.max_force)


# ================================================================
# REQUEST/RESPONSE MODELS
# ================================================================

class SingleRequest(BaseModel):
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


class SingleResponse(BaseModel):
    """Single prediction response"""
    force_N: float = Field(..., description="Control force (Newtons)")
    force_kN: float = Field(..., description="Control force (kilonewtons)")
    inference_time_ms: float = Field(..., description="Inference time (ms)")
    model: str = "Perfect RL (Champion)"


class BatchRequest(BaseModel):
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


class BatchResponse(BaseModel):
    """Batch prediction response"""
    forces_N: List[float] = Field(..., description="Control forces (Newtons)")
    forces_kN: List[float] = Field(..., description="Control forces (kilonewtons)")
    count: int = Field(..., description="Number of predictions")
    total_time_ms: float = Field(..., description="Total time (ms)")
    avg_time_ms: float = Field(..., description="Average time per prediction (ms)")
    model: str = "Perfect RL (Champion)"


# ================================================================
# FASTAPI APPLICATION
# ================================================================

# Create app
app = FastAPI(
    title="Perfect RL TMD Control API",
    description="🏆 Champion AI controller - Beats fuzzy logic by 14%",
    version="1.0.0"
)

# Global model instance
model = None


@app.on_event("startup")
async def startup():
    """Load model on startup"""
    global model
    model_path = "simple_rl_models/perfect_rl_final.zip"
    
    try:
        model = PerfectRLModel(model_path)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print(f"   Looking for: {model_path}")
        print(f"   Make sure model file exists!")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Perfect RL TMD Control API",
        "status": "🏆 Champion Model Online",
        "endpoints": {
            "single_prediction": "POST /predict",
            "batch_prediction": "POST /predict-batch",
            "health": "GET /health",
            "docs": "GET /docs"
        },
        "performance": {
            "TEST3_M4.5": "24.67 cm (21.8% vs passive)",
            "TEST4_M6.9": "20.80 cm (32% vs passive)",
            "average": "~21.5 cm across all scenarios",
            "vs_fuzzy": "+14% better"
        }
    }


@app.post("/predict", response_model=SingleResponse)
async def predict_single(request: SingleRequest):
    """
    Single prediction endpoint
    
    Returns control force for a single state.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start = time.time()
    
    try:
        force = model.predict_single(
            request.roof_displacement,
            request.roof_velocity,
            request.tmd_displacement,
            request.tmd_velocity
        )
        
        elapsed = (time.time() - start) * 1000
        
        return SingleResponse(
            force_N=float(force),
            force_kN=float(force / 1000),
            inference_time_ms=elapsed
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict-batch", response_model=BatchResponse)
async def predict_batch(request: BatchRequest):
    """
    Batch prediction endpoint
    
    Returns control forces for multiple states.
    More efficient than multiple single predictions.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate inputs
    n = len(request.roof_displacements)
    if not all(len(x) == n for x in [
        request.roof_velocities,
        request.tmd_displacements,
        request.tmd_velocities
    ]):
        raise HTTPException(
            status_code=422,
            detail="All arrays must have same length"
        )
    
    start = time.time()
    
    try:
        forces = model.predict_batch(
            request.roof_displacements,
            request.roof_velocities,
            request.tmd_displacements,
            request.tmd_velocities
        )
        
        elapsed = (time.time() - start) * 1000
        
        return BatchResponse(
            forces_N=forces.tolist(),
            forces_kN=(forces / 1000).tolist(),
            count=n,
            total_time_ms=elapsed,
            avg_time_ms=elapsed / n if n > 0 else 0
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


@app.get("/health")
async def health():
    """Health check endpoint"""
    if model is None:
        return {
            "status": "unhealthy",
            "model_loaded": False,
            "message": "Model not loaded"
        }
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "model": "Perfect RL (Champion)",
        "performance": "24.67 cm (TEST3), 20.80 cm (TEST4)",
        "message": "Ready for predictions"
    }


@app.get("/info")
async def info():
    """Model information"""
    return {
        "name": "Perfect RL (Champion)",
        "type": "SAC (Soft Actor-Critic)",
        "training": "Curriculum learning (50→100→150 kN)",
        "performance": {
            "TEST3_M4.5": "24.67 cm (21.8% vs passive)",
            "TEST4_M6.9": "20.80 cm (32% vs passive)",
            "TEST5_M6.7": "20.80 cm",
            "TEST6b_noise": "21.11 cm (+1.5% degradation)",
            "TEST6c_latency": "20.80 cm (0% degradation)",
            "TEST6d_dropout": "20.89 cm (+0.4% degradation)",
            "average": "~21.5 cm"
        },
        "comparison": {
            "vs_passive": "+32% better (on TEST4)",
            "vs_fuzzy": "+14% better (average)",
            "vs_rl_baseline": "+6% better (TEST3)",
            "rank": "🥇 1st place out of 5 methods"
        },
        "force_range": "±150 kN",
        "avg_force": "~85 kN",
        "robustness": "Excellent (< 3% degradation)"
    }


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("  PERFECT RL API SERVER")
    print("="*70)
    print("\n🏆 Champion model: Beats fuzzy logic by 14%")
    print("   Performance: 24.67 cm (TEST3), 20.80 cm (TEST4)")
    print("\n📊 Endpoints:")
    print("   POST /predict        - Single prediction")
    print("   POST /predict-batch  - Batch prediction")
    print("   GET  /health         - Health check")
    print("   GET  /info           - Model info")
    print("   GET  /docs           - Interactive docs")
    print("\n🚀 Starting server...")
    print("   URL: http://localhost:8000")
    print("   Docs: http://localhost:8000/docs")
    print("   Health: http://localhost:8000/health")
    print("\n" + "="*70 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

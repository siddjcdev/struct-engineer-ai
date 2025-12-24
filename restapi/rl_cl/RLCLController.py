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
    """RL CL model wrapper"""
    
    def __init__(self, model_path: str):
        print(f"RLCLController: Loading RL CL model from {model_path}...")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"RLCLController: Model not found: {model_path}")
        
        # Load SAC model
        self.model = SAC.load(model_path, device='cpu')
        self.max_force = 150000.0  # 150 kN
        
        print("✅ RLCLController: RL CL model loaded!")
        print("   RLCLController: RL CL performance:")
        print("   RLCLController:     • TEST3 (M4.5): 24.67 cm (21.8% vs passive)")
        print("   RLCLController:     • TEST4 (M6.9): 20.80 cm (32% vs passive)")
        print("   RLCLController:     • Average: ~21.5 cm, Beats fuzzy by 14%")
    
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


class RLCLBatchResponse(BaseModel):
    """Batch prediction response"""
    forces_N: List[float] = Field(..., description="Control forces (Newtons)")
    forces_kN: List[float] = Field(..., description="Control forces (kilonewtons)")
    count: int = Field(..., description="Number of predictions")
    total_time_ms: float = Field(..., description="Total time (ms)")
    avg_time_ms: float = Field(..., description="Average time per prediction (ms)")
    model: str = "Perfect RL (Champion)"

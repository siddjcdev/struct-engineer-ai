from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import List, Optional
from pydantic import BaseModel,Field
import json
import shutil
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import time
import sys
import os



from fuzzy.fixed_fuzzy_controller import FixedFuzzyTMDController as fuzzy, FuzzyBatchRequest, FuzzyBatchResponse


from models.tmd_models import (
    TMDSimulation,
    BaselinePerformance,
    TMDResults,
    TMDConfiguration,
    Improvements,
    TimeSeriesData,
    PerformanceComparison,
    InputData,
    BuildingState,
    ControlOutput
)

from rl_baseline.rl_controller import RLTMDController, RLSingleRequest, RLSingleResponse,RLBatchRequest, RLBatchResponse

from rl_cl.RLCLController import (
    RLCLBatchRequest, RLCLBatchResponse, RLCLController,
    RLCLSingleRequest, RLCLSingleResponse,
    RLCLSimulationRequest, RLCLSimulationResponse
)

# ============================================================================
# FastAPI App
# ============================================================================
app = FastAPI(
    title="TMD Simulation API with RL Inferences",
    description="REST API for Tuned Mass Damper (TMD) simulation data with Reinforcement Learning inferences",
    version="2.0.0"
)

# --- PATHS (Deployment-ready) ---
ROOT_PATH = Path(__file__).parent

# Global variables
DATA_FILE = Path("data/simulation.json")
#MODEL_FILE = (Path(__file__).parent.parent / "neuralnet" / "src" / "models" / "tmd_trained_model_peer.pth").resolve()
RL_MODEL_FILE = Path("models/rl_model.zip")
RL_CL_MODEL_FILE = Path("models/rl_cl_model_final.zip")

# For deployment: Use environment variable if available, otherwise use a relative path
MATLAB_OUTPUT_DIR_ENV = '' #os.getenv('MATLAB_OUTPUT_DIR', None)
if MATLAB_OUTPUT_DIR_ENV:
    MATLAB_OUTPUT_DIR = Path(MATLAB_OUTPUT_DIR_ENV)
else:
    # For local development, use the original path
    # For deployment, this will just be skipped
    MATLAB_OUTPUT_DIR = Path(r"data/matlab_outputs").resolve()
FUZZY_OUTPUT_DIR = ROOT_PATH / "data" / "fuzzy_outputs"
FUZZY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

simulation_data: Optional[TMDSimulation] = None

# Initialize comprehensive fuzzy controller
fuzzy_controller = fuzzy(
    # displacement_range=(-0.5, 0.5),    # ±50 cm
    # velocity_range=(-2.0, 2.0),         # ±2 m/s
    # force_range=(-100000, 100000)       # ±100 kN
)

# Debug: print resolved path
print("MAIN: Resolved DATA_FILE path:", DATA_FILE.resolve())
print(f"MAIN: Resolved MATLAB output directory: {MATLAB_OUTPUT_DIR}")

print(f"MAIN: RL Model file path: {RL_MODEL_FILE}")
print(f"MAIN: RL Model file exists: {RL_MODEL_FILE.exists()}")

print(f"MAIN: RL CL Model file path: {RL_CL_MODEL_FILE}")
print(f"MAIN: RL CL Model file exists: {RL_CL_MODEL_FILE.exists()}")
# ============================================================================
# Startup Functions
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    print("="*70)
    print("MAIN: Starting Fuzzy API with RL Inferences")
    print("="*70)
    #print(f"✅ Force Range: ±{fuzzy_controller.force_range[1]/1000:.1f} kN")
    print(f"✅ MAIN: Fuzzy Output Directory: {FUZZY_OUTPUT_DIR}")
    load_RL_model()
    print("\n" + "="*70)
    print("✅ MAIN: Startup complete, API is ready to serve requests")
    print("="*70)

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_RL_model():
    """Load trained RL model"""
    global rl_controller, rl_cl_controller
    try:
        # At startup, initialize RL controller
        rl_controller = RLTMDController(str(RL_MODEL_FILE))

        rl_cl_controller = RLCLController(str(RL_CL_MODEL_FILE))
        print("✅ MAIN: RL models ready for inference")
        return True
    except FileNotFoundError:
        print(f"⚠️  MAIN: Warning: {RL_MODEL_FILE} not found. Reinforcement learning endpoints will be disabled.")
        print(f"   MAIN: Train the model first: python train_perfect_rl_simple.py")
        rl_cl_controller = None
        rl_controller = None
        return False
    except Exception as e:
        print(f"❌ MAIN: Error loading RL model: {e}")
        rl_controller = None
        rl_cl_controller = None
        return False


# ============================================================================
# General Endpoints
# ============================================================================
@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information"""

    print("\n📊 Endpoints:")

    return {
        "message": "TMD Simulation API with Neural Network Inference",
        "version": "2.0.0",
        "status": "ready",
        "fuzzy_controller": "active",
        "simulation_data": "available" if simulation_data else "not_loaded",
        "endpoints": {
            "general": {
                "health": "/health",
                "docs": "/docs",
            },
            "fuzzy": {
                "fuzzy_control": "/fuzzy/predict-batch",
                "fuzzy_batch": "/fuzzy/test",
                "fuzzy_stats": "/fuzzy/stats",
            },
            "RL-baseline": {
                "rl_single": "POST /rl/predict",
                "rl_batch": "POST /rl/predict-batch",
                "rl_status": "GET /rl/health",
            },
            "RL-CL": {
                "rl_single": "POST /rl-cl/predict",
                "rl_batch": "POST /rl-cl/predict-batch",
                "rl_simulate": "POST /rl-cl/simulate (with full metrics)",
                "rl_status": "GET /rl-cl/health",
                "rl_info": "GET /rl-cl/info",
            },
        },
        "performance": {
            "TEST3_M4.5": "24.67 cm (21.8% vs passive)",
            "TEST4_M6.9": "20.80 cm (32% vs passive)",
            "average": "~21.5 cm across all scenarios",
            "vs_fuzzy": "+14% better"
        }
    }


@app.get("/health", tags=["General"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "fuzzy_controller": "active",
        "data_loaded": simulation_data is not None,
        "rl_model_loaded": rl_controller is not None,
        "rl_cl_model_loaded": rl_cl_controller is not None

    }


# ============================================================================
# FUZZY LOGIC CONTROL ENDPOINTS (PRIMARY)
# ============================================================================

@app.post("/fuzzy/predict-batch", tags=["Fuzzy Control"], response_model=FuzzyBatchResponse)
async def fuzzy_batch_predict(request: FuzzyBatchRequest):
    """
    Batch fuzzy logic TMD control
    
    Returns forces in kN (kilonewtons)
    """
    import time
    start_time = time.time()
    
    try:
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
        
        # Compute forces (returns Newtons)
        forces_N = fuzzy_controller.compute_batch(
            request.roof_displacements,
            request.roof_velocities,
            request.tmd_displacements,
            request.tmd_velocities
        )
        
        # Convert to kN
        forces_kN = forces_N / 1000.0
        
        # Calculate inference time
        inference_time = (time.time() - start_time) * 1000  # ms
        
        return FuzzyBatchResponse(
            forces=forces_kN.tolist(),
            force_unit="kN",
            num_predictions=len(forces_kN),
            inference_time_ms=inference_time
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Fuzzy controller error: {str(e)}"
        )


@app.get("/fuzzy/test", tags=["Fuzzy Control"])
async def test_fuzzy_controller():
    """
    Test endpoint to verify fuzzy controller is working
    """
    
    # Test case: TMD at 0.15m right, moving 0.8 m/s right
    # Should return negative force (push left)
    test_disp = 0.15
    test_vel = 0.8
    
    force_N = fuzzy_controller.compute(test_disp, test_vel)
    print(f"TEST: Fuzzy controller output force: {force_N} N")

    force_kN = force_N / 1000.0

    test_status = "PASS" if force_kN < 0 else "FAIL"
    print(f"TEST: Fuzzy controller test status: {test_status}")
    
    return {
        "status": "ok",
        "test_input": {
            "relative_displacement": test_disp,
            "relative_velocity": test_vel
        },
        "output": {
            "force_N": force_N,
            "force_kN": force_kN
        },
        "expected": "Negative force (push left)",
        "correct": test_status
    }


@app.get("/fuzzy/stats", tags=["Fuzzy Control"])
async def get_fuzzy_stats():
    """Get fuzzy controller statistics and configuration"""
    return fuzzy_controller.get_stats()



# ================================================================
# REINFORCEMENT LEARNING (BASELINE) CONTROL ENDPOINTS
# ================================================================

# Endpoint

@app.post("/rl/predict", response_model=RLSingleResponse, tags=["Reinforcement Learning"])
async def predict_single(request: RLSingleRequest):
    """
    RL Single prediction endpoint
    
    Returns control force for a single state.
    """
    if rl_controller is None:
        raise HTTPException(status_code=503, detail="RL Model not loaded")
    
    start = time.time()
    
    try:
        force = rl_controller.predict(
            request.roof_displacement,
            request.roof_velocity,
            request.tmd_displacement,
            request.tmd_velocity
        )
        
        elapsed = (time.time() - start) * 1000
        
        return RLSingleResponse(
            force_N=float(force),
            force_kN=float(force / 1000),
            inference_time_ms=elapsed
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RL Prediction error: {str(e)}")





@app.post("/rl/predict-batch", tags=["Reinforcement Learning"], response_model=RLBatchResponse)
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
        print(f"RL BATCH: Predicted forces (N): {forces_N}")
        
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


@app.get("/rl/test", tags=["Reinforcement Learning"])
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


# ============================================================================
# REINFORCEMENT LEARNING (CURRICULUM LEARNING) CONTROL ENDPOINTS
# ============================================================================
@app.get("/rl-cl/test", tags=["Reinforcement Learning CL"])
async def test_rl_cl_controller():
    # Test with sample state
    test_state = {
        "roof_disp": 0.15,
        "roof_vel": 0.8,
        "tmd_disp": 0.16,
        "tmd_vel": 0.9
    }
    
    force_N = rl_cl_controller.predict_single(
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


@app.post("/rl-cl/predict", response_model=RLCLSingleResponse, tags=["Reinforcement Learning CL"])
async def predict_single(request: RLCLSingleRequest):
    """
    RL Single prediction endpoint
    
    Returns control force for a single state.
    """
    if rl_cl_controller is None:
        raise HTTPException(status_code=503, detail="RL Model not loaded")
    
    start = time.time()
    
    try:
        force = rl_cl_controller.predict_single(
            request.roof_displacement,
            request.roof_velocity,
            request.tmd_displacement,
            request.tmd_velocity
        )
        
        elapsed = (time.time() - start) * 1000
        
        return RLCLSingleResponse(
            force_N=float(force),
            force_kN=float(force / 1000),
            inference_time_ms=elapsed
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RL Prediction error: {str(e)}")


@app.post("/rl-cl/predict-batch", response_model=RLCLBatchResponse, tags=["Reinforcement Learning CL"])
async def predict_batch(request: RLCLBatchRequest):
    """
    Batch prediction endpoint
    
    Returns control forces for multiple states.
    More efficient than multiple single predictions.
    """
    if rl_cl_controller is None:
        raise HTTPException(status_code=503, detail="RL Model not loaded")
    
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
        forces = rl_cl_controller.predict_batch(
            request.roof_displacements,
            request.roof_velocities,
            request.tmd_displacements,
            request.tmd_velocities
        )
        
        elapsed = (time.time() - start) * 1000
        
        return RLCLBatchResponse(
            forces_N=forces.tolist(),
            forces_kN=(forces / 1000).tolist(),
            count=n,
            total_time_ms=elapsed,
            avg_time_ms=elapsed / n if n > 0 else 0
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


@app.get("/rl-cl/status", tags=["Reinforcement Learning CL"])
async def health():
    """Health check endpoint"""
    if rl_cl_controller is None:
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

@app.get("/rl-cl/info", tags=["Reinforcement Learning CL"])
async def info():
    """RL Model information"""
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


@app.post("/rl-cl/simulate", response_model=RLCLSimulationResponse, tags=["Reinforcement Learning CL"])
async def simulate(request: RLCLSimulationRequest):
    """
    Run full earthquake simulation with RL-CL controller

    Returns all performance metrics including:
    - RMS of roof displacement
    - Peak roof displacement
    - Maximum interstory drift
    - DCR (Drift Concentration Ratio)
    - Peak and mean control forces
    """
    if rl_cl_controller is None:
        raise HTTPException(status_code=503, detail="RL-CL Model not loaded")

    import numpy as np

    # Validate inputs
    if len(request.earthquake_data) == 0:
        raise HTTPException(
            status_code=422,
            detail="Earthquake data cannot be empty"
        )

    start = time.time()

    try:
        # Run simulation
        metrics = rl_cl_controller.simulate_episode(
            earthquake_data=np.array(request.earthquake_data),
            dt=request.dt
        )

        elapsed = (time.time() - start) * 1000

        return RLCLSimulationResponse(
            forces_N=metrics['forces_N'],
            forces_kN=metrics['forces_kN'],
            count=len(metrics['forces_N']),
            rms_roof_displacement=metrics['rms_roof_displacement'],
            peak_roof_displacement=metrics['peak_roof_displacement'],
            max_drift=metrics['max_drift'],
            DCR=metrics['DCR'],
            peak_force=metrics['peak_force'],
            mean_force=metrics['mean_force'],
            peak_force_kN=metrics['peak_force_kN'],
            mean_force_kN=metrics['mean_force_kN'],
            simulation_time_ms=elapsed
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation error: {str(e)}")

# ============================================================================
# MAIN ENTRY POINT  
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*70)
    print("TMD Fuzzy API with RL Inferences")
    print("="*70)
    print("\n🏆 RL CL model: Beats fuzzy logic by 14%")
    print("   Performance: 24.67 cm (TEST3), 20.80 cm (TEST4)")
    print("="*70)
    print("Starting server on http://0.0.0.0:8080")
    print("API Documentation: http://0.0.0.0:8080/docs")
    print("="*70 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="debug")
"""
FASTAPI ENDPOINT FOR FIXED FUZZY CONTROLLER
===========================================

Add this to your main.py FastAPI application

This replaces your old fuzzy endpoint with the corrected version
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
from fixed_fuzzy_controller import FixedFuzzyTMDController

# Initialize the fixed fuzzy controller (do this once at startup)
fuzzy_controller = FixedFuzzyTMDController()


# ================================================================
# REQUEST/RESPONSE MODELS
# ================================================================

class FuzzyBatchRequest(BaseModel):
    """
    Batch prediction request for fuzzy controller
    
    IMPORTANT: These should be RELATIVE values!
    - displacements: TMD_disp - roof_disp (in meters)
    - velocities: TMD_vel - roof_vel (in m/s)
    """
    displacements: List[float]
    velocities: List[float]


class FuzzyBatchResponse(BaseModel):
    """Batch prediction response"""
    forces: List[float]  # Forces in kN
    force_unit: str = "kN"
    num_predictions: int
    inference_time_ms: float


# ================================================================
# ENDPOINTS
# ================================================================

@app.post("/fuzzylogic-batch", response_model=FuzzyBatchResponse)
async def fuzzy_batch_predict(request: FuzzyBatchRequest):
    """
    Batch fuzzy logic TMD control
    
    Returns forces in kN (kilonewtons)
    """
    import time
    start_time = time.time()
    
    try:
        # Validate input
        if len(request.displacements) != len(request.velocities):
            raise HTTPException(
                status_code=422,
                detail="Displacements and velocities must have same length"
            )
        
        if len(request.displacements) == 0:
            raise HTTPException(
                status_code=422,
                detail="Empty input arrays"
            )
        
        # Compute forces (returns Newtons)
        forces_N = fuzzy_controller.compute_batch(
            request.displacements,
            request.velocities
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


@app.get("/fuzzylogic/test")
async def test_fuzzy_controller():
    """
    Test endpoint to verify fuzzy controller is working
    """
    
    # Test case: TMD at 0.15m right, moving 0.8 m/s right
    # Should return negative force (push left)
    test_disp = 0.15
    test_vel = 0.8
    
    force_N = fuzzy_controller.compute(test_disp, test_vel)
    force_kN = force_N / 1000.0
    
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
        "correct": force_kN < 0
    }


# ================================================================
# HEALTH CHECK (update to include fuzzy)
# ================================================================

@app.get("/health")
async def health_check():
    """Enhanced health check including fuzzy controller"""
    
    return {
        "status": "healthy",
        "fuzzy_controller": "loaded",
        "nn_model_loaded": True,  # If you have NN
        "endpoints": [
            "/fuzzylogic-batch",
            "/fuzzylogic/test",
            "/nn/predict-batch",
            "/health"
        ]
    }
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import List, Optional
from pydantic import BaseModel
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import time
from models.tmd_models import (
    TMDSimulation,
    BaselinePerformance,
    TMDResults,
    TMDConfiguration,
    Improvements,
    TimeSeriesData,
    PerformanceComparison,
    InputData
)


# ============================================================================
# Neural Network Model Definition
# ============================================================================
class TMDNeuralNetwork(nn.Module):
    """Neural Network for TMD Control - must match training architecture"""
    
    def __init__(self):
        super(TMDNeuralNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        return self.network(x)


class NeuralTMDController:
    """Neural network controller for real-time TMD force prediction"""
    
    def __init__(self, model_path: str):
        self.model = TMDNeuralNetwork()
        self.normalization = None
        self.device = torch.device('cpu')  # Use CPU for deployment
        
        # Load model
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.normalization = checkpoint['normalization']
        
        # Set to evaluation mode
        self.model.eval()
        
        # Convert normalization to numpy for faster computation
        self.input_mean = np.array(self.normalization['input_mean'], dtype=np.float32)
        self.input_std = np.array(self.normalization['input_std'], dtype=np.float32)
        self.output_mean = np.array(self.normalization['output_mean'], dtype=np.float32)
        self.output_std = np.array(self.normalization['output_std'], dtype=np.float32)
        
        print(f"✅ Neural network model loaded from {model_path}")
    
    def compute_single(self, displacement: float, velocity: float) -> float:
        """Compute control force for a single state"""
        x = np.array([[displacement, velocity]], dtype=np.float32)
        x_norm = (x - self.input_mean) / self.input_std
        
        with torch.no_grad():
            x_tensor = torch.tensor(x_norm, dtype=torch.float32)
            y_norm = self.model(x_tensor).numpy()
        
        y = y_norm * self.output_std + self.output_mean
        return float(y[0, 0])
    
    def compute_batch(self, displacements: np.ndarray, velocities: np.ndarray) -> np.ndarray:
        """Compute control forces for multiple states (vectorized - much faster!)"""
        X = np.column_stack([displacements, velocities]).astype(np.float32)
        X_norm = (X - self.input_mean) / self.input_std
        
        with torch.no_grad():
            X_tensor = torch.tensor(X_norm, dtype=torch.float32)
            y_norm = self.model(X_tensor).numpy()
        
        y = y_norm * self.output_std + self.output_mean
        return y.flatten()


# ============================================================================
# Pydantic Models for Neural Network Endpoints
# ============================================================================
class PredictSingleRequest(BaseModel):
    """Request model for single prediction"""
    displacement: float  # meters
    velocity: float      # m/s


class PredictBatchRequest(BaseModel):
    """Request model for batch predictions"""
    displacements: List[float]  # meters
    velocities: List[float]     # m/s


class PredictSingleResponse(BaseModel):
    """Response model for single prediction"""
    force: float
    force_unit: str
    inference_time_ms: float


class PredictBatchResponse(BaseModel):
    """Response model for batch predictions"""
    forces: List[float]
    force_unit: str
    n_predictions: int
    inference_time_ms: float
    time_per_prediction_ms: float


# ============================================================================
# FastAPI App
# ============================================================================
app = FastAPI(
    title="TMD Simulation API",
    description="REST API for Tuned Mass Damper (TMD) simulation data with Neural Network inference",
    version="2.0.0"
)

# Global variables
DATA_FILE = Path("data/simulation.json")
#MODEL_FILE = (Path(__file__).parent.parent / "neuralnet" / "src" / "models" / "tmd_trained_model_peer.pth").resolve()
MODEL_FILE = Path("models/tmd_trained_model_peer.pth")

simulation_data: Optional[TMDSimulation] = None
nn_controller: Optional[NeuralTMDController] = None

# Debug: print resolved path
#C:\Dev\dAmpIng26\git\struct-engineer-ai\neuralnet\src\models\tmd_trained_model_peer.pth
print(f"Model file path: {MODEL_FILE}")
print(f"Model file exists: {MODEL_FILE.exists()}")

# ============================================================================
# Startup Functions
# ============================================================================
def load_simulation_data():
    """Load simulation data from JSON file"""
    global simulation_data
    try:
        with open(DATA_FILE, 'r') as f:
            data = json.load(f)
            simulation_data = TMDSimulation(**data)
            print("✅ Simulation data loaded successfully.")
    except FileNotFoundError:
        print(f"⚠️  Warning: {DATA_FILE} not found. Simulation endpoints will return empty responses.")
        simulation_data = None
    except Exception as e:
        print(f"❌ Error loading simulation data: {e}")
        simulation_data = None


def load_neural_network():
    """Load trained neural network model"""
    global nn_controller
    try:
        nn_controller = NeuralTMDController(str(MODEL_FILE))
        print("✅ Neural network controller ready for inference")
        return True
    except FileNotFoundError:
        print(f"⚠️  Warning: {MODEL_FILE} not found. Neural network endpoints will be disabled.")
        print(f"   Train the model first: python train_neural_network_peer.py")
        nn_controller = None
        return False
    except Exception as e:
        print(f"❌ Error loading neural network: {e}")
        nn_controller = None
        return False


@app.on_event("startup")
async def startup_event():
    """Load data and model on startup"""
    print("="*70)
    print("Starting TMD Simulation API with Neural Network Inference")
    print("="*70)
    load_simulation_data()
    load_neural_network()
    print("="*70)
    print("✅ API Ready")
    print("="*70)


# ============================================================================
# General Endpoints
# ============================================================================
@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "TMD Simulation API with Neural Network Inference",
        "version": "2.0.0",
        "endpoints": {
            "simulation": "/simulation",
            "baseline": "/baseline",
            "tmd_results": "/tmd-results",
            "tmd_config": "/tmd-config",
            "improvements": "/improvements",
            "comparison": "/comparison",
            "time_series": "/time-series",
            "input": "/input",
            "nn_predict_single": "/nn/predict",
            "nn_predict_batch": "/nn/predict-batch",
            "nn_status": "/nn/status"
        }
    }


@app.get("/health", tags=["General"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "data_loaded": simulation_data is not None,
        "nn_model_loaded": nn_controller is not None
    }


# ============================================================================
# Neural Network Inference Endpoints
# ============================================================================
@app.get("/nn/status", tags=["Neural Network"])
async def nn_status():
    """Check neural network model status"""
    if nn_controller is None:
        return {
            "model_loaded": False,
            "error": "Model not loaded. Please train the model first.",
            "model_path": str(MODEL_FILE)
        }
    
    return {
        "model_loaded": True,
        "model_path": str(MODEL_FILE),
        "normalization": nn_controller.normalization,
        "device": str(nn_controller.device)
    }


@app.post("/nn/predict", response_model=PredictSingleResponse, tags=["Neural Network"])
async def nn_predict_single(request: PredictSingleRequest):
    """
    Predict TMD control force for a single building state
    
    This endpoint computes the optimal control force based on the current
    roof displacement and velocity using the trained neural network.
    
    **Input:**
    - displacement: Building roof displacement in meters
    - velocity: Building roof velocity in m/s
    
    **Output:**
    - force: Control force in kilonewtons (kN)
    - inference_time_ms: Time taken for prediction
    
    **Example for MATLAB:**
    ```matlab
    % Single prediction
    url = 'http://localhost:8000/nn/predict';
    data = struct('displacement', 0.15, 'velocity', 0.8);
    options = weboptions('RequestMethod', 'post', 'MediaType', 'application/json');
    response = webread(url, data, options);
    force = response.force;  % Control force in kN
    ```
    """
    if nn_controller is None:
        raise HTTPException(
            status_code=503,
            detail="Neural network model not loaded. Please train the model first."
        )
    
    try:
        start_time = time.time()
        force = nn_controller.compute_single(request.displacement, request.velocity)
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return PredictSingleResponse(
            force=float(force),
            force_unit="kN",
            inference_time_ms=inference_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/nn/predict-batch", response_model=PredictBatchResponse, tags=["Neural Network"])
async def nn_predict_batch(request: PredictBatchRequest):
    """
    Predict TMD control forces for multiple building states (BATCH MODE - FASTER!)
    
    This endpoint is optimized for time series data and is MUCH faster than
    making individual predictions. Use this for earthquake simulation data.
    
    **Input:**
    - displacements: Array of roof displacements in meters
    - velocities: Array of roof velocities in m/s
    
    **Output:**
    - forces: Array of control forces in kilonewtons (kN)
    - n_predictions: Number of predictions made
    - inference_time_ms: Total time for all predictions
    - time_per_prediction_ms: Average time per prediction
    
    **Example for MATLAB:**
    ```matlab
    % Batch prediction (MUCH FASTER for time series!)
    url = 'http://localhost:8000/nn/predict-batch';
    data = struct('displacements', roof_disp, 'velocities', roof_vel);
    options = weboptions('RequestMethod', 'post', 'MediaType', 'application/json');
    response = webread(url, data, options);
    control_forces = response.forces;  % Array of forces in kN
    
    % Convert kN to N for simulation
    control_forces_N = control_forces * 1000;
    ```
    """
    if nn_controller is None:
        raise HTTPException(
            status_code=503,
            detail="Neural network model not loaded. Please train the model first."
        )
    
    try:
        # Validate input lengths match
        if len(request.displacements) != len(request.velocities):
            raise HTTPException(
                status_code=400,
                detail="Displacement and velocity arrays must have the same length"
            )
        
        # Convert to numpy arrays
        displacements = np.array(request.displacements, dtype=np.float32)
        velocities = np.array(request.velocities, dtype=np.float32)
        
        # Compute predictions (vectorized - very fast!)
        start_time = time.time()
        forces = nn_controller.compute_batch(displacements, velocities)
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        
        n_predictions = len(forces)
        time_per_prediction = inference_time / n_predictions if n_predictions > 0 else 0
        
        return PredictBatchResponse(
            forces=forces.tolist(),
            force_unit="kN",
            n_predictions=n_predictions,
            inference_time_ms=inference_time,
            time_per_prediction_ms=time_per_prediction
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# ============================================================================
# Original Simulation Endpoints (unchanged)
# ============================================================================
@app.get("/simulation", response_model=TMDSimulation, tags=["Simulation"])
async def get_simulation():
    """Get complete simulation data"""
    if simulation_data is None:
        raise HTTPException(status_code=404, detail="Simulation data not found")
    return simulation_data


@app.get("/simulation/metadata", tags=["Simulation"])
async def get_metadata():
    """Get simulation metadata"""
    if simulation_data is None:
        raise HTTPException(status_code=404, detail="Simulation data not found")
    return {
        "version": simulation_data.version,
        "timestamp": simulation_data.timestamp,
        "metadata": simulation_data.metadata,
        "v7": simulation_data.v7
    }


@app.get("/baseline", response_model=BaselinePerformance, tags=["Performance"])
async def get_baseline():
    """Get baseline performance metrics (without TMD)"""
    if simulation_data is None:
        raise HTTPException(status_code=404, detail="Simulation data not found")
    return simulation_data.baseline


@app.get("/tmd-results", response_model=TMDResults, tags=["Performance"])
async def get_tmd_results():
    """Get TMD performance results"""
    if simulation_data is None:
        raise HTTPException(status_code=404, detail="Simulation data not found")
    return simulation_data.tmd_results


@app.get("/tmd-config", response_model=TMDConfiguration, tags=["Performance"])
async def get_tmd_config():
    """Get TMD configuration parameters"""
    if simulation_data is None:
        raise HTTPException(status_code=404, detail="Simulation data not found")
    return simulation_data.tmd


@app.get("/improvements", response_model=Improvements, tags=["Performance"])
async def get_improvements():
    """Get performance improvements with TMD"""
    if simulation_data is None:
        raise HTTPException(status_code=404, detail="Simulation data not found")
    return simulation_data.improvements


@app.get("/comparison", response_model=List[PerformanceComparison], tags=["Performance"])
async def get_comparison():
    """Get side-by-side comparison of baseline vs TMD performance"""
    if simulation_data is None:
        raise HTTPException(status_code=404, detail="Simulation data not found")
    
    comparisons = [
        PerformanceComparison(
            metric="DCR",
            baseline=simulation_data.baseline.DCR,
            with_tmd=simulation_data.tmd_results.DCR,
            improvement_pct=simulation_data.improvements.dcr_reduction_pct,
            unit="ratio"
        ),
        PerformanceComparison(
            metric="Max Drift",
            baseline=simulation_data.baseline.max_drift,
            with_tmd=simulation_data.tmd_results.max_drift,
            improvement_pct=simulation_data.improvements.drift_reduction_pct,
            unit="m"
        ),
        PerformanceComparison(
            metric="Max Roof Displacement",
            baseline=simulation_data.baseline.max_roof,
            with_tmd=simulation_data.tmd_results.max_roof,
            improvement_pct=simulation_data.improvements.roof_reduction_pct,
            unit="m"
        ),
        PerformanceComparison(
            metric="RMS Displacement",
            baseline=simulation_data.baseline.rms_displacement,
            with_tmd=simulation_data.tmd_results.rms_displacement,
            improvement_pct=simulation_data.improvements.rms_disp_reduction_pct,
            unit="m"
        ),
        PerformanceComparison(
            metric="RMS Velocity",
            baseline=simulation_data.baseline.rms_velocity,
            with_tmd=simulation_data.tmd_results.rms_velocity,
            improvement_pct=simulation_data.improvements.rms_vel_reduction_pct,
            unit="m/s"
        ),
        PerformanceComparison(
            metric="RMS Acceleration",
            baseline=simulation_data.baseline.rms_acceleration,
            with_tmd=simulation_data.tmd_results.rms_acceleration,
            improvement_pct=simulation_data.improvements.rms_acc_reduction_pct,
            unit="m/s²"
        )
    ]
    
    return comparisons


@app.get("/time-series", response_model=TimeSeriesData, tags=["Time Series"])
async def get_time_series(
    start_time: Optional[float] = Query(None, description="Start time in seconds"),
    end_time: Optional[float] = Query(None, description="End time in seconds")
):
    """Get time series data with optional time range filtering"""
    if simulation_data is None:
        raise HTTPException(status_code=404, detail="Simulation data not found")
    
    time_series = simulation_data.time_series
    
    # If no filtering, return all data
    if start_time is None and end_time is None:
        return time_series
    
    # Filter by time range
    filtered_indices = []
    for i, t in enumerate(time_series.time):
        if start_time is not None and t < start_time:
            continue
        if end_time is not None and t > end_time:
            break
        filtered_indices.append(i)
    
    return TimeSeriesData(
        time=[time_series.time[i] for i in filtered_indices],
        earthquake_acceleration=[time_series.earthquake_acceleration[i] for i in filtered_indices],
        baseline_roof=[time_series.baseline_roof[i] for i in filtered_indices],
        tmd_roof=[time_series.tmd_roof[i] for i in filtered_indices]
    )


@app.get("/time-series/summary", tags=["Time Series"])
async def get_time_series_summary():
    """Get summary statistics of time series data"""
    if simulation_data is None:
        raise HTTPException(status_code=404, detail="Simulation data not found")
    
    ts = simulation_data.time_series
    
    return {
        "total_points": len(ts.time),
        "duration": ts.time[-1] - ts.time[0],
        "time_step": ts.time[1] - ts.time[0] if len(ts.time) > 1 else 0,
        "earthquake": {
            "max_acceleration": max(ts.earthquake_acceleration),
            "min_acceleration": min(ts.earthquake_acceleration)
        },
        "baseline_roof": {
            "max_displacement": max(ts.baseline_roof),
            "min_displacement": min(ts.baseline_roof)
        },
        "tmd_roof": {
            "max_displacement": max(ts.tmd_roof),
            "min_displacement": min(ts.tmd_roof)
        }
    }


@app.get("/input", response_model=InputData, tags=["Input"])
async def get_input():
    """Get simulation input parameters (earthquake and wind data)"""
    if simulation_data is None:
        raise HTTPException(status_code=404, detail="Simulation data not found")
    return simulation_data.input


@app.get("/dcr-profile", tags=["Performance"])
async def get_dcr_profile():
    """Get DCR profile comparison by floor"""
    if simulation_data is None:
        raise HTTPException(status_code=404, detail="Simulation data not found")
    
    n_floors = len(simulation_data.baseline.dcr_profile)
    
    return {
        "floors": list(range(1, n_floors + 1)),
        "baseline_dcr": simulation_data.baseline.dcr_profile,
        "tmd_dcr": simulation_data.tmd_results.dcr_profile,
        "tmd_floor": simulation_data.tmd.floor
    }


@app.post("/reload", tags=["General"])
async def reload_data():
    """Reload simulation data and neural network model from files"""
    load_simulation_data()
    nn_loaded = load_neural_network()
    return {
        "status": "success",
        "message": "Data reloaded",
        "data_loaded": simulation_data is not None,
        "nn_model_loaded": nn_loaded
    }


if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*70)
    print("TMD Simulation API with Neural Network Inference")
    print("="*70)
    print("Starting server on http://localhost:8080")
    print("API Documentation: http://localhost:8080/docs")
    print("="*70 + "\n")
    uvicorn.run(app, host="localhost", port=8080)
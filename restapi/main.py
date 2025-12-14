from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import List, Optional
from pydantic import BaseModel
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

import fuzzy.FuzzyTMD
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

class FuzzySingleRequest(BaseModel):
    """Request model for single prediction"""
    displacement: float  # meters
    velocity: float      # m/s

class FuzzyBatchRequest(BaseModel):
    """Request model for batch predictions"""
    displacements: List[float]  # meters, "Array of displacements (m)"
    velocities: List[float]     # m/s, "Array of velocities (m/s)"
    #accelerations: Optional[List[float]]  # m/s², "Array of accelerations (m/s²)"

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

# --- PATHS (Deployment-ready) ---
ROOT_PATH = Path(__file__).parent
# For deployment: Use environment variable if available, otherwise use a relative path
MATLAB_OUTPUT_DIR_ENV = os.getenv('MATLAB_OUTPUT_DIR', None)
if MATLAB_OUTPUT_DIR_ENV:
    MATLAB_OUTPUT_DIR = Path(MATLAB_OUTPUT_DIR_ENV)
else:
    # For local development, use the original path
    # For deployment, this will just be skipped
    MATLAB_OUTPUT_DIR = Path(r"C:\Dev\dAmpIng26\shared_data\results")
FUZZY_OUTPUT_DIR = ROOT_PATH / "data" / "fuzzy_outputs"
FUZZY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

simulation_data: Optional[TMDSimulation] = None
nn_controller: Optional[NeuralTMDController] = None

# Initialize comprehensive fuzzy controller
fuzzy_controller = fuzzy.FuzzyTMD.FuzzyTMDController(
    displacement_range=(-0.5, 0.5),    # ±50 cm
    velocity_range=(-2.0, 2.0),         # ±2 m/s
    force_range=(-100000, 100000)       # ±100 kN
)

# Debug: print resolved path
#C:\Dev\dAmpIng26\git\struct-engineer-ai\neuralnet\src\models\tmd_trained_model_peer.pth
print(f"Model file path: {MODEL_FILE}")
print(f"Model file exists: {MODEL_FILE.exists()}")

# ============================================================================
# Startup Functions
# ============================================================================





@app.on_event("startup")
async def startup_event():
    """Load data and model on startup"""
    print("="*70)
    print("Starting TMD Simulation API with Neural Network Inference")
    print("="*70)
    load_simulation_data()
    load_neural_network()
    print("\n" + "="*70)
    print("TMD FUZZY CONTROL API - READY FOR DEPLOYMENT")
    print("="*70)
    print(f"✅ Fuzzy Controller: Active")
    print(f"✅ Force Range: ±{fuzzy_controller.force_range[1]/1000:.1f} kN")
    print(f"✅ Output Directory: {FUZZY_OUTPUT_DIR}")
    print(f"{'✅' if simulation_data else '⚠️ '} Simulation Data: {'Loaded' if simulation_data else 'Not Available'}")
    print("="*70 + "\n")
    print("="*70)
    print("✅ API Ready")
    print("="*70)

    
    
# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def _update_latest_simulation():
    """Find the latest MATLAB simulation file and copy it to data/simulation.json"""
    
    # Skip MATLAB file loading on deployment
    if not MATLAB_OUTPUT_DIR.exists():
        print("⚠️  MATLAB output directory not found. Skipping simulation data load.")
        print("   (This is normal for deployment - fuzzy controller will still work!)")
        return True
        
    if not MATLAB_OUTPUT_DIR.is_dir():
        print(f"⚠️  Not a directory: {MATLAB_OUTPUT_DIR}")
        return False

    search_dir = MATLAB_OUTPUT_DIR
    pattern = "tmd_v7_simulation_*.json"
    
    try:
        simulation_files = list(search_dir.glob(pattern))
        
        if not simulation_files:
            print(f"Warning: No simulation files found in '{search_dir}'. Using existing data.")
            return True

        latest_file = max(simulation_files, key=lambda p: p.stat().st_mtime)
        
        DATA_FILE.parent.mkdir(exist_ok=True)
        
        shutil.copy2(latest_file, DATA_FILE)
        print(f"Successfully updated '{DATA_FILE.name}' with data from '{latest_file.name}'")
        return True

    except Exception as e:
        print(f"An unexpected error occurred during simulation file update: {e}")
        return False


def load_simulation_data():
    """Update and load simulation data from JSON file"""
    global simulation_data
    
    success = _update_latest_simulation()
    
    if not success and not DATA_FILE.exists():
        print("⚠️  No simulation data available. Fuzzy controller will still work!")
        return

    try:
        with open(DATA_FILE, 'r') as f:
            data = json.load(f)
            simulation_data = TMDSimulation(**data)
            print("✅ Simulation data loaded successfully.")
    except FileNotFoundError:
        print(f"⚠️  {DATA_FILE} not found. Simulation endpoints will return 404.")
        print("   Fuzzy control endpoints will still work!")
        simulation_data = None
    except Exception as e:
        print(f"Error loading simulation data: {e}")
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

# ============================================================================
# General Endpoints
# ============================================================================
@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "TMD Simulation API with Neural Network Inference",
        "version": "2.0.0",
        "status": "ready",
        "fuzzy_controller": "active",
        "simulation_data": "available" if simulation_data else "not_loaded",
        "endpoints": {
            "fuzzy_control": "/fuzzylogic",
            "fuzzy_batch": "/fuzzylogic-batch",
            "fuzzy_stats": "/fuzzy-stats",
            "simulation": "/simulation",
            "health": "/health",
            "docs": "/docs",
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
        "timestamp": datetime.now().isoformat(),
        "fuzzy_controller": "active",
        "computations": fuzzy_controller.computation_count,
        "data_loaded": simulation_data is not None,
        "nn_model_loaded": nn_controller is not None
    }


# ============================================================================
# FUZZY LOGIC CONTROL ENDPOINTS (PRIMARY)
# ============================================================================

@app.get("/fuzzylogic", tags=["Fuzzy Control"])
async def fuzzy_logic_control_from_simulation():
    """
    GET variant of /fuzzylogic — reads RMS inputs from data/simulation.json and runs the same controller.
    Expects fields named (any of):
      - rms_displacement, rms_velocity, rms_acceleration
    or nested under keys like 'baseline', 'metrics' or 'rms'.
    """
    if not DATA_FILE.exists():
        raise HTTPException(status_code=404, detail=f"Simulation file not found: {DATA_FILE}")

    try:
        with open(DATA_FILE, 'r') as f:
            sim = json.load(f)

        print("Extracting RMS values from simulation data...")
        print("Available keys:", list(sim.keys()))
        print("Searching for: rms_displacement, rms_velocity, [rms_acceleration]")
        def _extract(key):
            # direct
            if key in sim:
                return sim[key]
            print(f"Key '{key}' not found directly in simulation data.")
            # common containers
            container = "tmd_results"
            print(f"Checking container '{container}' for key '{key}'...")
            if container in sim and isinstance(sim[container], dict) and key in sim[container]:
                return sim[container][key]
            print(f"Key '{key}' not found in containers {('tmd_results',)} or directly in simulation data.")
            # try short name inside 'rms' container
            short = key.replace("rms_", "")
            if "rms" in sim and isinstance(sim["rms"], dict) and short in sim["rms"]:
                return sim["rms"][short]
            return None

        displacement = _extract("rms_displacement")
        velocity = _extract("rms_velocity")
        acceleration = _extract("rms_acceleration")

        if displacement is None or velocity is None:
            raise HTTPException(
                status_code=400,
                detail="Required RMS fields not found in simulation.json (rms_displacement, rms_velocity[, rms_acceleration])"
            )

        # run controller (reuse same compute & saving pattern as POST)
        control_force = fuzzy_controller.compute(float(displacement), float(velocity), float(acceleration) if acceleration is not None else None)

        response = {
            "timestamp": datetime.now().isoformat(),
            "source": str(DATA_FILE),
            "computation_number": fuzzy_controller.computation_count,
            "inputs": {
                "displacement_m": displacement,
                "velocity_ms": velocity,
                "acceleration_ms2": acceleration
            },
            "output": {
                "control_force_N": control_force,
                "control_force_kN": control_force / 1000,
                "direction": "left (negative)" if control_force < 0 else "right (positive)"
            },
            "controller_info": {
                "type": "comprehensive_fuzzy_logic",
                "displacement_range_m": fuzzy_controller.displacement_range,
                "velocity_range_ms": fuzzy_controller.velocity_range,
                "force_range_kN": [fuzzy_controller.force_range[0]/1000, fuzzy_controller.force_range[1]/1000]
            }
        }

        # Save result files
        output_filename = f"fuzzy_simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_path = FUZZY_OUTPUT_DIR / output_filename
        with open(output_path, 'w') as f:
            json.dump(response, f, indent=2)
        latest_path = FUZZY_OUTPUT_DIR / "fuzzy_output_latest.json"
        with open(latest_path, 'w') as f:
            json.dump(response, f, indent=2)

        response["saved_to"] = str(output_path)
        response["latest_file"] = str(latest_path)

        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running fuzzy controller from simulation.json: {e}")

# ...existing code...

@app.post("/fuzzylogic", tags=["Fuzzy Control"])
async def fuzzy_logic_control(
    request: FuzzySingleRequest
):
    """
    PRIMARY FUZZY LOGIC ENDPOINT
    
    Receives displacement and velocity, returns control force
    Automatically saves to JSON file
    
    Example: /fuzzylogic?displacement=0.1&velocity=0.5
    """
    try:
        # Compute control force
        control_force = fuzzy_controller.compute(request.displacement, request.velocity, None)
        
        # Prepare response
        response = {
            "timestamp": datetime.now().isoformat(),
            "computation_number": fuzzy_controller.computation_count,
            "inputs": {
                "displacement_m": request.displacement,
                "velocity_ms": request.velocity,
                "acceleration_ms2": None
            },
            "output": {
                "control_force_N": control_force,
                "control_force_kN": control_force / 1000,
                "direction": "left (negative)" if control_force < 0 else "right (positive)"
            },
            "controller_info": {
                "type": "comprehensive_fuzzy_logic",
                "displacement_range_m": fuzzy_controller.displacement_range,
                "velocity_range_ms": fuzzy_controller.velocity_range,
                "force_range_kN": [fuzzy_controller.force_range[0]/1000, 
                                   fuzzy_controller.force_range[1]/1000]
            }
        }
        
        # Save to JSON file
        output_filename = f"fuzzy_output_{fuzzy_controller.computation_count:06d}.json"
        output_path = FUZZY_OUTPUT_DIR / output_filename
        
        with open(output_path, 'w') as f:
            json.dump(response, f, indent=2)
        
        # Also save as "latest"
        latest_path = FUZZY_OUTPUT_DIR / "fuzzy_output_latest.json"
        with open(latest_path, 'w') as f:
            json.dump(response, f, indent=2)
        
        response["saved_to"] = str(output_path)
        response["latest_file"] = str(latest_path)
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fuzzy computation error: {str(e)}")


@app.post("/fuzzylogic-batch", tags=["Fuzzy Control"])
async def fuzzy_logic_batch(request: FuzzyBatchRequest):
    """
    Batch prediction using fuzzy logic controller
    Matches the format of /nn/predict-batch endpoint
    """
    if fuzzy_controller is None:
        raise HTTPException(
            status_code=503,
            detail="Fuzzy logic controller not loaded. Please initialize the controller first."
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
        
        # Option 1: If your fuzzy controller only needs displacement & velocity
        forces_N = fuzzy_controller.compute_batch(displacements, velocities)
        
        # Option 2: If your fuzzy controller requires accelerations parameter (even if None)
        # Uncomment this and comment out Option 1 above if needed:
        # forces_N = fuzzy_controller.compute_batch(displacements, velocities, None)
        
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Convert forces from N to kN (to match NN endpoint)
        forces_kN = forces_N / 1000.0
        
        n_predictions = len(forces_kN)
        time_per_prediction = inference_time / n_predictions if n_predictions > 0 else 0
        
        return PredictBatchResponse(
            forces=forces_kN.tolist(),
            force_unit="kN",
            n_predictions=n_predictions,
            inference_time_ms=inference_time,
            time_per_prediction_ms=time_per_prediction
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")



@app.get("/fuzzy-stats", tags=["Fuzzy Control"])
async def get_fuzzy_stats():
    """Get fuzzy controller statistics and configuration"""
    return fuzzy_controller.get_stats()


@app.get("/fuzzy-history", tags=["Fuzzy Control"])
async def get_fuzzy_history(
    last_n: int = Query(100, description="Number of recent computations to return")
):
    """Get recent fuzzy computation history"""
    history = fuzzy_controller.computation_history[-last_n:]
    
    return {
        "total_computations": fuzzy_controller.computation_count,
        "returned_count": len(history),
        "history": history
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
    url = 'http://0.0.0.0:8000/nn/predict';
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
    url = 'http://0.0.0.0:8000/nn/predict-batch';
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
    print("Starting server on http://0.0.0.0:8080")
    print("API Documentation: http://0.0.0.0:8080/docs")
    print("="*70 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8080)
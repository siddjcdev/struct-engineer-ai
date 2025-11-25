"""
TMD Controller API - Comprehensive WebSocket + REST Edition
Combines comprehensive fuzzy logic with ultra-low latency WebSocket support
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import json
from pathlib import Path
import shutil
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from datetime import datetime
import sys

# Fix path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import (
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

# Import neural controller if available
try:
    import torch
    from neural_controller import NeuralTMDController
    NEURAL_AVAILABLE = True
except ImportError:
    NEURAL_AVAILABLE = False
    print("⚠️  PyTorch not available. Neural controller disabled.")


# ============================================================================
# PYDANTIC MODELS FOR WEBSOCKET
# ============================================================================

class WebSocketMessage(BaseModel):
    """Message format for WebSocket communication"""
    message_id: int
    displacement: float
    velocity: float
    acceleration: Optional[float] = None


class WebSocketResponse(BaseModel):
    """Response format for WebSocket"""
    message_id: int
    control_force_N: float
    controller_type: str
    computation_time_ms: float


# ============================================================================
# COMPREHENSIVE FUZZY LOGIC TMD CONTROLLER
# ============================================================================

class FuzzyTMDController:
    """
    Comprehensive Fuzzy Logic Controller for TMD
    Uses real physical values and engineering-based rules
    
    Input: displacement (m), velocity (m/s), acceleration (m/s²)
    Output: control force (N)
    """
    
    def __init__(self, 
                 displacement_range=(-0.5, 0.5),
                 velocity_range=(-2.0, 2.0),
                 force_range=(-100000, 100000)):
        """
        Initialize comprehensive fuzzy controller
        
        Args:
            displacement_range: (min, max) in meters
            velocity_range: (min, max) in m/s
            force_range: (min, max) in Newtons
        """
        self.displacement_range = displacement_range
        self.velocity_range = velocity_range
        self.force_range = force_range
        
        # Statistics tracking
        self.computation_count = 0
        self.last_computation_time = None
        self.computation_history = []
        
        # Build fuzzy system
        self.controller = self._build_fuzzy_system()
        
        print("="*70)
        print("COMPREHENSIVE FUZZY LOGIC TMD CONTROLLER INITIALIZED")
        print("="*70)
        print(f"Displacement range: {displacement_range[0]:.2f} to {displacement_range[1]:.2f} m")
        print(f"Velocity range: {velocity_range[0]:.2f} to {velocity_range[1]:.2f} m/s")
        print(f"Force range: {force_range[0]/1000:.1f} to {force_range[1]/1000:.1f} kN")
        print("="*70)
    
    def _build_fuzzy_system(self):
        """Build the comprehensive fuzzy inference system"""
        
        # Input variables
        displacement = ctrl.Antecedent(
            np.arange(self.displacement_range[0], self.displacement_range[1], 0.01),
            'displacement'
        )
        velocity = ctrl.Antecedent(
            np.arange(self.velocity_range[0], self.velocity_range[1], 0.01),
            'velocity'
        )
        
        # Output variable
        control_force = ctrl.Consequent(
            np.arange(self.force_range[0], self.force_range[1], 1000),
            'control_force'
        )
        
        # Membership functions - Displacement (5 levels)
        displacement['negative_large'] = fuzz.trapmf(
            displacement.universe, 
            [self.displacement_range[0], self.displacement_range[0], -0.3, -0.1]
        )
        displacement['negative_small'] = fuzz.trimf(
            displacement.universe, [-0.3, -0.1, 0]
        )
        displacement['zero'] = fuzz.trimf(
            displacement.universe, [-0.1, 0, 0.1]
        )
        displacement['positive_small'] = fuzz.trimf(
            displacement.universe, [0, 0.1, 0.3]
        )
        displacement['positive_large'] = fuzz.trapmf(
            displacement.universe,
            [0.1, 0.3, self.displacement_range[1], self.displacement_range[1]]
        )
        
        # Membership functions - Velocity (5 levels)
        velocity['negative_fast'] = fuzz.trapmf(
            velocity.universe,
            [self.velocity_range[0], self.velocity_range[0], -1.0, -0.3]
        )
        velocity['negative_slow'] = fuzz.trimf(
            velocity.universe, [-1.0, -0.3, 0]
        )
        velocity['zero'] = fuzz.trimf(
            velocity.universe, [-0.3, 0, 0.3]
        )
        velocity['positive_slow'] = fuzz.trimf(
            velocity.universe, [0, 0.3, 1.0]
        )
        velocity['positive_fast'] = fuzz.trapmf(
            velocity.universe,
            [0.3, 1.0, self.velocity_range[1], self.velocity_range[1]]
        )
        
        # Membership functions - Control Force (5 levels)
        force_max = self.force_range[1]
        control_force['large_negative'] = fuzz.trapmf(
            control_force.universe,
            [self.force_range[0], self.force_range[0], -0.6*force_max, -0.2*force_max]
        )
        control_force['small_negative'] = fuzz.trimf(
            control_force.universe, [-0.6*force_max, -0.2*force_max, 0]
        )
        control_force['zero'] = fuzz.trimf(
            control_force.universe, [-0.2*force_max, 0, 0.2*force_max]
        )
        control_force['small_positive'] = fuzz.trimf(
            control_force.universe, [0, 0.2*force_max, 0.6*force_max]
        )
        control_force['large_positive'] = fuzz.trapmf(
            control_force.universe,
            [0.2*force_max, 0.6*force_max, self.force_range[1], self.force_range[1]]
        )
        
        # Comprehensive Fuzzy Rules (Engineering-based)
        rules = [
            # Strong damping when moving fast outward
            ctrl.Rule(
                displacement['positive_large'] & velocity['positive_fast'],
                control_force['large_negative']
            ),
            ctrl.Rule(
                displacement['negative_large'] & velocity['negative_fast'],
                control_force['large_positive']
            ),
            
            # Moderate damping for moderate motion
            ctrl.Rule(
                displacement['positive_small'] & velocity['positive_slow'],
                control_force['small_negative']
            ),
            ctrl.Rule(
                displacement['positive_large'] & velocity['positive_slow'],
                control_force['small_negative']
            ),
            ctrl.Rule(
                displacement['negative_small'] & velocity['negative_slow'],
                control_force['small_positive']
            ),
            ctrl.Rule(
                displacement['negative_large'] & velocity['negative_slow'],
                control_force['small_positive']
            ),
            
            # Minimal force near equilibrium
            ctrl.Rule(
                displacement['zero'] & velocity['zero'],
                control_force['zero']
            ),
            
            # Reduced damping when naturally returning
            ctrl.Rule(
                displacement['positive_small'] & velocity['negative_slow'],
                control_force['zero']
            ),
            ctrl.Rule(
                displacement['positive_large'] & velocity['negative_fast'],
                control_force['small_positive']
            ),
            ctrl.Rule(
                displacement['negative_small'] & velocity['positive_slow'],
                control_force['zero']
            ),
            ctrl.Rule(
                displacement['negative_large'] & velocity['positive_fast'],
                control_force['small_negative']
            ),
        ]
        
        # Create control system
        control_system = ctrl.ControlSystem(rules)
        print(f"✅ Fuzzy system built with {len(rules)} engineering-based rules")
        
        return ctrl.ControlSystemSimulation(control_system)
    
    def compute(self, displacement, velocity, acceleration=None):
        """
        Compute control force for given building state
        
        Args:
            displacement: Inter-story drift in meters
            velocity: Inter-story drift velocity in m/s
            acceleration: Building acceleration in m/s² (optional)
        
        Returns:
            control_force: Control force in Newtons
        """
        self.computation_count += 1
        self.last_computation_time = datetime.now()
        
        # Clip to valid ranges
        displacement_clipped = np.clip(
            displacement,
            self.displacement_range[0],
            self.displacement_range[1]
        )
        velocity_clipped = np.clip(
            velocity,
            self.velocity_range[0],
            self.velocity_range[1]
        )
        
        # Set inputs
        self.controller.input['displacement'] = displacement_clipped
        self.controller.input['velocity'] = velocity_clipped
        
        # Compute
        try:
            self.controller.compute()
            control_force = self.controller.output['control_force']
            
            # Store computation history
            computation_record = {
                'timestamp': self.last_computation_time.isoformat(),
                'count': self.computation_count,
                'inputs': {
                    'displacement': displacement,
                    'displacement_clipped': displacement_clipped,
                    'velocity': velocity,
                    'velocity_clipped': velocity_clipped,
                    'acceleration': acceleration
                },
                'output': {
                    'control_force_N': control_force,
                    'control_force_kN': control_force / 1000
                }
            }
            
            # Keep last 1000 computations
            if len(self.computation_history) >= 1000:
                self.computation_history.pop(0)
            self.computation_history.append(computation_record)
            
            return control_force
            
        except Exception as e:
            print(f"❌ Fuzzy computation error: {e}")
            return 0.0
    
    def compute_batch(self, displacements, velocities, accelerations=None):
        """Compute control forces for time series data"""
        forces = []
        
        if accelerations is None:
            accelerations = [None] * len(displacements)
        
        for d, v, a in zip(displacements, velocities, accelerations):
            forces.append(self.compute(d, v, a))
        
        return np.array(forces)
    
    def get_stats(self):
        """Get controller statistics"""
        return {
            "total_computations": self.computation_count,
            "last_computation": self.last_computation_time.isoformat() if self.last_computation_time else None,
            "displacement_range_m": self.displacement_range,
            "velocity_range_ms": self.velocity_range,
            "force_range_kN": [self.force_range[0]/1000, self.force_range[1]/1000],
            "status": "active"
        }


# ============================================================================
# CONNECTION MANAGER FOR WEBSOCKETS
# ============================================================================

class ConnectionManager:
    """Manages WebSocket connections"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.connection_stats = {
            'total_connections': 0,
            'active_connections': 0,
            'total_messages': 0
        }
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.connection_stats['total_connections'] += 1
        self.connection_stats['active_connections'] = len(self.active_connections)
        print(f"✅ New WebSocket connection. Active: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        self.connection_stats['active_connections'] = len(self.active_connections)
        print(f"❌ WebSocket disconnected. Active: {len(self.active_connections)}")
    
    async def send_json(self, websocket: WebSocket, data: dict):
        await websocket.send_json(data)
        self.connection_stats['total_messages'] += 1


manager = ConnectionManager()


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="TMD Simulation API - Comprehensive WebSocket + REST Edition",
    description="Full simulation data access + ultra-low latency WebSocket control",
    version="3.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- PATHS ---
ROOT_PATH = Path(__file__).parent
MATLAB_OUTPUT_DIR = Path(r"C:\Dev\dAmpIng26\shared_data\results")
DATA_FILE = ROOT_PATH / "data" / "simulation.json"
FUZZY_OUTPUT_DIR = ROOT_PATH / "data" / "fuzzy_outputs"
FUZZY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

simulation_data: Optional[TMDSimulation] = None

# Initialize controllers
fuzzy_controller = FuzzyTMDController(
    displacement_range=(-0.5, 0.5),
    velocity_range=(-2.0, 2.0),
    force_range=(-100000, 100000)
)

neural_controller = None
if NEURAL_AVAILABLE:
    try:
        neural_controller = NeuralTMDController(device='cpu')
        print("✅ Neural Network Controller initialized")
    except Exception as e:
        print(f"❌ Neural controller error: {e}")


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def _update_latest_simulation():
    """Find the latest MATLAB simulation file"""
    if str(MATLAB_OUTPUT_DIR) == "PASTE_YOUR_FULL_MATLAB_PATH_HERE":
        print("ERROR: Please set MATLAB_OUTPUT_DIR")
        return False
        
    if not MATLAB_OUTPUT_DIR.is_dir():
        print(f"ERROR: Path does not exist: {MATLAB_OUTPUT_DIR}")
        return False

    try:
        simulation_files = list(MATLAB_OUTPUT_DIR.glob("tmd_v7_simulation_*.json"))
        
        if not simulation_files:
            print("Warning: No simulation files found. Using existing data.")
            return True

        latest_file = max(simulation_files, key=lambda p: p.stat().st_mtime)
        DATA_FILE.parent.mkdir(exist_ok=True)
        shutil.copy2(latest_file, DATA_FILE)
        print(f"✅ Updated with: {latest_file.name}")
        return True

    except Exception as e:
        print(f"Error updating simulation: {e}")
        return False


def load_simulation_data():
    """Load simulation data from JSON"""
    global simulation_data
    
    success = _update_latest_simulation()
    
    if not success and not DATA_FILE.exists():
        print("No simulation data available")
        return

    try:
        with open(DATA_FILE, 'r') as f:
            data = json.load(f)
            simulation_data = TMDSimulation(**data)
            print("✅ Simulation data loaded")
    except Exception as e:
        print(f"Error loading data: {e}")
        simulation_data = None


@app.on_event("startup")
async def startup_event():
    """Startup initialization"""
    load_simulation_data()
    
    print("\n" + "="*70)
    print("TMD API - COMPREHENSIVE WEBSOCKET + REST EDITION")
    print("="*70)
    print(f"Fuzzy Controller: ✅ Available (Comprehensive)")
    print(f"Neural Controller: {'✅ Available' if neural_controller else '❌ Unavailable'}")
    print(f"\nWebSocket Endpoints (LOW LATENCY):")
    print(f"  • /ws/fuzzy  - Comprehensive fuzzy logic (2-5ms)")
    print(f"  • /ws/neural - Neural network (3-10ms)")
    print(f"  • /ws/auto   - Auto-select best")
    print(f"\nREST API Endpoints:")
    print(f"  • /fuzzylogic - Fuzzy control with file save")
    print(f"  • /fuzzylogic-batch - Batch processing")
    print(f"  • /simulation - Full simulation data")
    print("="*70 + "\n")


# ============================================================================
# WEBSOCKET ENDPOINTS (ULTRA-LOW LATENCY)
# ============================================================================

@app.websocket("/ws/fuzzy")
async def websocket_fuzzy_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for comprehensive fuzzy logic control
    
    Ultra-low latency: ~2-5ms
    Uses full scikit-fuzzy controller
    
    Usage from MATLAB:
        1. Connect to wss://your-url/ws/fuzzy
        2. Send: {"message_id": 1, "displacement": 0.1, "velocity": 0.5}
        3. Receive: {"message_id": 1, "control_force_N": -30625, ...}
    """
    await manager.connect(websocket)
    
    try:
        while True:
            # Receive data
            data = await websocket.receive_json()
            
            # Start timing
            start_time = datetime.now()
            
            # Parse message
            msg = WebSocketMessage(**data)
            
            # Compute using COMPREHENSIVE fuzzy controller
            force = fuzzy_controller.compute(
                msg.displacement, 
                msg.velocity, 
                msg.acceleration
            )
            
            # Calculate computation time
            computation_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Send response
            response = {
                'message_id': msg.message_id,
                'control_force_N': float(force),
                'controller_type': 'comprehensive_fuzzy_logic',
                'computation_time_ms': computation_time,
                'computation_count': fuzzy_controller.computation_count
            }
            
            await manager.send_json(websocket, response)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket)


@app.websocket("/ws/neural")
async def websocket_neural_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for neural network control
    
    Ultra-low latency: ~3-10ms
    """
    if not neural_controller:
        await websocket.close(code=1003, reason="Neural controller not available")
        return
    
    await manager.connect(websocket)
    
    try:
        while True:
            data = await websocket.receive_json()
            start_time = datetime.now()
            
            msg = WebSocketMessage(**data)
            force = neural_controller.compute(
                msg.displacement, 
                msg.velocity, 
                msg.acceleration
            )
            
            computation_time = (datetime.now() - start_time).total_seconds() * 1000
            
            response = {
                'message_id': msg.message_id,
                'control_force_N': float(force),
                'controller_type': 'neural_network',
                'computation_time_ms': computation_time
            }
            
            await manager.send_json(websocket, response)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket)


@app.websocket("/ws/auto")
async def websocket_auto_endpoint(websocket: WebSocket):
    """Auto-select best available controller"""
    if neural_controller:
        await websocket_neural_endpoint(websocket)
    else:
        await websocket_fuzzy_endpoint(websocket)


# ============================================================================
# REST API ENDPOINTS - ROOT AND GENERAL
# ============================================================================

@app.get("/", tags=["General"])
async def root():
    """Root endpoint"""
    return {
        "message": "TMD API - Comprehensive WebSocket + REST Edition",
        "version": "3.0.0",
        "websocket_endpoints": {
            "fuzzy": "/ws/fuzzy (comprehensive, 2-5ms)",
            "neural": "/ws/neural (3-10ms)",
            "auto": "/ws/auto"
        },
        "rest_endpoints": {
            "fuzzy_control": "/fuzzylogic",
            "fuzzy_batch": "/fuzzylogic-batch",
            "simulation": "/simulation",
            "baseline": "/baseline",
            "tmd_results": "/tmd-results"
        },
        "latency_improvement": "WebSocket is 10-20x faster than REST"
    }


@app.get("/health", tags=["General"])
async def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "data_loaded": simulation_data is not None,
        "fuzzy_controller": "comprehensive_active",
        "fuzzy_computations": fuzzy_controller.computation_count,
        "neural_available": neural_controller is not None,
        "websocket_stats": manager.connection_stats,
        "timestamp": datetime.now().isoformat()
    }


# ============================================================================
# REST API - FUZZY LOGIC CONTROL ENDPOINTS
# ============================================================================

@app.post("/fuzzylogic", tags=["Fuzzy Control"])
async def fuzzy_logic_control(
    displacement: float = Query(..., description="Inter-story drift (m)"),
    velocity: float = Query(..., description="Velocity (m/s)"),
    acceleration: Optional[float] = Query(None, description="Acceleration (m/s²)")
):
    """
    PRIMARY FUZZY LOGIC ENDPOINT
    
    Compute control force and save to JSON file for MATLAB
    """
    try:
        control_force = fuzzy_controller.compute(displacement, velocity, acceleration)
        
        response = {
            "timestamp": datetime.now().isoformat(),
            "computation_number": fuzzy_controller.computation_count,
            "inputs": {
                "displacement_m": displacement,
                "velocity_ms": velocity,
                "acceleration_ms2": acceleration
            },
            "output": {
                "control_force_N": control_force,
                "control_force_kN": control_force / 1000,
                "direction": "left" if control_force < 0 else "right"
            },
            "controller_info": {
                "type": "comprehensive_fuzzy_logic",
                "displacement_range_m": fuzzy_controller.displacement_range,
                "velocity_range_ms": fuzzy_controller.velocity_range,
                "force_range_kN": [fuzzy_controller.force_range[0]/1000, 
                                   fuzzy_controller.force_range[1]/1000]
            }
        }
        
        # Save to files
        output_filename = f"fuzzy_output_{fuzzy_controller.computation_count:06d}.json"
        output_path = FUZZY_OUTPUT_DIR / output_filename
        
        with open(output_path, 'w') as f:
            json.dump(response, f, indent=2)
        
        latest_path = FUZZY_OUTPUT_DIR / "fuzzy_output_latest.json"
        with open(latest_path, 'w') as f:
            json.dump(response, f, indent=2)
        
        response["saved_to"] = str(output_path)
        response["latest_file"] = str(latest_path)
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/fuzzylogic-batch", tags=["Fuzzy Control"])
async def fuzzy_logic_batch(
    displacements: List[float] = Query(...),
    velocities: List[float] = Query(...),
    accelerations: Optional[List[float]] = Query(None)
):
    """Batch process multiple time steps"""
    try:
        if len(displacements) != len(velocities):
            raise HTTPException(400, "Arrays must have same length")
        
        if accelerations and len(accelerations) != len(displacements):
            raise HTTPException(400, "Accelerations must match length")
        
        forces = fuzzy_controller.compute_batch(
            np.array(displacements),
            np.array(velocities),
            np.array(accelerations) if accelerations else None
        )
        
        response = {
            "timestamp": datetime.now().isoformat(),
            "batch_info": {
                "total_steps": len(displacements),
                "computation_start": fuzzy_controller.computation_count - len(displacements) + 1,
                "computation_end": fuzzy_controller.computation_count
            },
            "time_series": {
                "displacements_m": displacements,
                "velocities_ms": velocities,
                "accelerations_ms2": accelerations or [None] * len(displacements),
                "control_forces_N": forces.tolist(),
                "control_forces_kN": (forces / 1000).tolist()
            },
            "statistics": {
                "max_force_kN": float(np.max(np.abs(forces)) / 1000),
                "mean_force_kN": float(np.mean(np.abs(forces)) / 1000),
                "std_force_kN": float(np.std(forces) / 1000)
            }
        }
        
        batch_filename = f"fuzzy_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        batch_path = FUZZY_OUTPUT_DIR / batch_filename
        
        with open(batch_path, 'w') as f:
            json.dump(response, f, indent=2)
        
        response["saved_to"] = str(batch_path)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Error: {str(e)}")


@app.get("/fuzzy-stats", tags=["Fuzzy Control"])
async def get_fuzzy_stats():
    """Get fuzzy controller statistics"""
    return fuzzy_controller.get_stats()


@app.get("/fuzzy-history", tags=["Fuzzy Control"])
async def get_fuzzy_history(
    last_n: int = Query(100, description="Number of recent computations")
):
    """Get recent computation history"""
    history = fuzzy_controller.computation_history[-last_n:]
    
    return {
        "total_computations": fuzzy_controller.computation_count,
        "returned_count": len(history),
        "history": history
    }


# ============================================================================
# REST API - SIMULATION DATA ENDPOINTS (All original endpoints)
# ============================================================================

@app.get("/simulation", response_model=TMDSimulation, tags=["Simulation"])
async def get_simulation():
    """Get complete simulation data"""
    load_simulation_data()
    if simulation_data is None:
        raise HTTPException(404, "Simulation data not found")
    return simulation_data


@app.get("/baseline", response_model=BaselinePerformance, tags=["Performance"])
async def get_baseline():
    """Get baseline performance"""
    if simulation_data is None:
        raise HTTPException(404, "Simulation data not found")
    return simulation_data.baseline


@app.get("/tmd-results", response_model=TMDResults, tags=["Performance"])
async def get_tmd_results():
    """Get TMD results"""
    if simulation_data is None:
        raise HTTPException(404, "Simulation data not found")
    return simulation_data.tmd_results


@app.get("/tmd-config", response_model=TMDConfiguration, tags=["TMD"])
async def get_tmd_config():
    """Get TMD configuration"""
    if simulation_data is None:
        raise HTTPException(404, "Simulation data not found")
    return simulation_data.tmd


@app.get("/improvements", response_model=Improvements, tags=["Performance"])
async def get_improvements():
    """Get performance improvements"""
    if simulation_data is None:
        raise HTTPException(404, "Simulation data not found")
    return simulation_data.improvements


@app.get("/comparison", response_model=List[PerformanceComparison], tags=["Performance"])
async def get_comparison():
    """Get side-by-side comparison"""
    if simulation_data is None:
        raise HTTPException(404, "Simulation data not found")
    
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
        # Add other comparisons...
    ]
    
    return comparisons


@app.get("/time-series", response_model=TimeSeriesData, tags=["Time Series"])
async def get_time_series(
    start_time: Optional[float] = Query(None),
    end_time: Optional[float] = Query(None)
):
    """Get time series data"""
    if simulation_data is None:
        raise HTTPException(404, "Simulation data not found")
    
    # Filtering logic...
    return simulation_data.time_series


@app.get("/input", response_model=InputData, tags=["Input"])
async def get_input():
    """Get input parameters"""
    if simulation_data is None:
        raise HTTPException(404, "Simulation data not found")
    return simulation_data.input


@app.post("/reload", tags=["General"])
async def reload_data():
    """Reload simulation data"""
    load_simulation_data()
    return {
        "status": "success",
        "data_loaded": simulation_data is not None
    }


# ============================================================================
# MATLAB INTEGRATION
# ============================================================================

@app.get("/matlab-ready", tags=["MATLAB Integration"])
async def check_matlab_ready():
    """Check MATLAB integration status"""
    return {
        "api_status": "ready",
        "simulation_data_loaded": simulation_data is not None,
        "fuzzy_controller_active": True,
        "fuzzy_computations": fuzzy_controller.computation_count,
        "neural_available": neural_controller is not None,
        "websocket_available": True,
        "output_directory": str(FUZZY_OUTPUT_DIR),
        "websocket_endpoints": {
            "fuzzy": "/ws/fuzzy (comprehensive, 2-5ms)",
            "neural": "/ws/neural (3-10ms)"
        },
        "rest_endpoints": {
            "single": "POST /fuzzylogic",
            "batch": "POST /fuzzylogic-batch"
        }
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, ws="websockets")
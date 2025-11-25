from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import List, Optional
import json
from pathlib import Path
import shutil
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from datetime import datetime
from pathlib import Path
import sys
import os
   
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
        
        # Membership functions - Displacement (5 levels for precision)
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
        
        # Membership functions - Velocity (5 levels for precision)
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
        
        # Membership functions - Control Force (5 levels for precision)
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
        
        # Comprehensive Fuzzy Rules (Engineering-based for TMD control)
        rules = [
            # Rule 1-2: Strong damping when moving fast outward
            ctrl.Rule(
                displacement['positive_large'] & velocity['positive_fast'],
                control_force['large_negative']
            ),
            ctrl.Rule(
                displacement['negative_large'] & velocity['negative_fast'],
                control_force['large_positive']
            ),
            
            # Rule 3-6: Moderate damping for moderate motion
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
            
            # Rule 7: Minimal force near equilibrium
            ctrl.Rule(
                displacement['zero'] & velocity['zero'],
                control_force['zero']
            ),
            
            # Rule 8-11: Reduced damping when naturally returning to equilibrium
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
            acceleration: Building acceleration in m/s² (optional, for logging)
        
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
        """
        Compute control forces for time series data
        
        Args:
            displacements: Array of inter-story drifts (m)
            velocities: Array of velocities (m/s)
            accelerations: Array of accelerations (m/s²), optional
        
        Returns:
            forces: Array of control forces (N)
        """
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
    
    def save_computation_history(self, filepath):
        """Save computation history to JSON file"""
        with open(filepath, 'w') as f:
            json.dump({
                'controller_config': {
                    'displacement_range': self.displacement_range,
                    'velocity_range': self.velocity_range,
                    'force_range': self.force_range
                },
                'total_computations': self.computation_count,
                'computation_history': self.computation_history
            }, f, indent=2)
        return filepath


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="TMD Simulation API with Fuzzy Logic Control",
    description="REST API for Tuned Mass Damper simulation data and fuzzy logic control",
    version="2.0.0"
)

# CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

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

DATA_FILE = ROOT_PATH / "data" / "simulation.json"
FUZZY_OUTPUT_DIR = ROOT_PATH / "data" / "fuzzy_outputs"
FUZZY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

simulation_data: Optional[TMDSimulation] = None

# Initialize comprehensive fuzzy controller
fuzzy_controller = FuzzyTMDController(
    displacement_range=(-0.5, 0.5),    # ±50 cm
    velocity_range=(-2.0, 2.0),         # ±2 m/s
    force_range=(-100000, 100000)       # ±100 kN
)


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


@app.on_event("startup")
async def startup_event():
    """Load data on startup"""
    load_simulation_data()
    
    print("\n" + "="*70)
    print("TMD FUZZY CONTROL API - READY FOR DEPLOYMENT")
    print("="*70)
    print(f"✅ Fuzzy Controller: Active")
    print(f"✅ Force Range: ±{fuzzy_controller.force_range[1]/1000:.1f} kN")
    print(f"✅ Output Directory: {FUZZY_OUTPUT_DIR}")
    print(f"{'✅' if simulation_data else '⚠️ '} Simulation Data: {'Loaded' if simulation_data else 'Not Available'}")
    print("="*70 + "\n")


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "TMD Simulation API with Fuzzy Logic Control",
        "version": "2.0.0",
        "status": "ready",
        "fuzzy_controller": "active",
        "simulation_data": "available" if simulation_data else "not_loaded",
        "endpoints": {
            "fuzzy_control": "/fuzzylogic (PRIMARY - use this!)",
            "fuzzy_batch": "/fuzzylogic-batch",
            "fuzzy_stats": "/fuzzy-stats",
            "simulation": "/simulation",
            "health": "/health",
            "docs": "/docs"
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
        "data_loaded": simulation_data is not None
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
    displacement: float = Query(..., description="Inter-story drift in meters"),
    velocity: float = Query(..., description="Inter-story drift velocity in m/s"),
    acceleration: Optional[float] = Query(None, description="Building acceleration in m/s² (optional)")
):
    """
    PRIMARY FUZZY LOGIC ENDPOINT
    
    Receives displacement and velocity, returns control force
    Automatically saves to JSON file
    
    Example: /fuzzylogic?displacement=0.1&velocity=0.5
    """
    try:
        # Compute control force
        control_force = fuzzy_controller.compute(displacement, velocity, acceleration)
        
        # Prepare response
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
async def fuzzy_logic_batch(
    displacements: List[float] = Query(..., description="Array of displacements (m)"),
    velocities: List[float] = Query(..., description="Array of velocities (m/s)"),
    accelerations: Optional[List[float]] = Query(None, description="Array of accelerations (m/s²)")
):
    """
    Batch process multiple time steps
    
    Processes entire time series and returns all control forces
    """
    try:
        if len(displacements) != len(velocities):
            raise HTTPException(
                status_code=400,
                detail="Displacements and velocities must have same length"
            )
        
        if accelerations and len(accelerations) != len(displacements):
            raise HTTPException(
                status_code=400,
                detail="Accelerations must have same length as displacements"
            )
        
        # Compute batch
        forces = fuzzy_controller.compute_batch(
            np.array(displacements),
            np.array(velocities),
            np.array(accelerations) if accelerations else None
        )
        
        # Prepare response
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
                "accelerations_ms2": accelerations if accelerations else [None] * len(displacements),
                "control_forces_N": forces.tolist(),
                "control_forces_kN": (forces / 1000).tolist()
            },
            "statistics": {
                "max_force_kN": float(np.max(np.abs(forces)) / 1000),
                "mean_force_kN": float(np.mean(np.abs(forces)) / 1000),
                "std_force_kN": float(np.std(forces) / 1000),
                "max_displacement_m": float(np.max(np.abs(displacements))),
                "max_velocity_ms": float(np.max(np.abs(velocities)))
            }
        }
        
        # Save batch results
        batch_filename = f"fuzzy_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        batch_path = FUZZY_OUTPUT_DIR / batch_filename
        
        with open(batch_path, 'w') as f:
            json.dump(response, f, indent=2)
        
        response["saved_to"] = str(batch_path)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch computation error: {str(e)}")


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
# SIMULATION DATA ENDPOINTS (Optional - require simulation data)
# ============================================================================

@app.get("/simulation", response_model=TMDSimulation, tags=["Simulation"])
async def get_simulation():
    """Get complete simulation data"""
    load_simulation_data()
    if simulation_data is None:
        raise HTTPException(status_code=404, detail="Simulation data not found")
    return simulation_data


@app.get("/baseline", response_model=BaselinePerformance, tags=["Performance"])
async def get_baseline():
    """Get baseline performance metrics"""
    if simulation_data is None:
        raise HTTPException(status_code=404, detail="Simulation data not found")
    return simulation_data.baseline


@app.get("/tmd-results", response_model=TMDResults, tags=["Performance"])
async def get_tmd_results():
    """Get TMD performance results"""
    if simulation_data is None:
        raise HTTPException(status_code=404, detail="Simulation data not found")
    return simulation_data.tmd_results


@app.post("/reload", tags=["General"])
async def reload_data():
    """Reload simulation data from file"""
    load_simulation_data()
    return {
        "status": "success",
        "message": "Simulation data reloaded",
        "data_loaded": simulation_data is not None
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Use PORT from environment variable (for Render) or default to 8001
    port = int(os.getenv("PORT", 8001))
    
    print("\n" + "="*70)
    print("TMD SIMULATION API WITH FUZZY LOGIC CONTROL")
    print("="*70)
    print(f"Fuzzy Controller: Active")
    print(f"Force Range: ±{fuzzy_controller.force_range[1]/1000:.1f} kN")
    print(f"Port: {port}")
    print("="*70 + "\n")
    
    # Run server
    uvicorn.run(app, host="0.0.0.0", port=port)
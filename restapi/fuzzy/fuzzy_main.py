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

# --- PATHS ---
ROOT_PATH = Path(__file__).parent

# !!! USER ACTION REQUIRED !!!
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
# DATA LOADING FUNCTIONS (Original from your code)
# ============================================================================

def _update_latest_simulation():
    """Find the latest MATLAB simulation file and copy it to data/simulation.json"""
    
    if str(MATLAB_OUTPUT_DIR) == "PASTE_YOUR_FULL_MATLAB_PATH_HERE":
        print("="*50)
        print("ERROR: Please edit 'main.py' and set the 'MATLAB_OUTPUT_DIR' variable.")
        print("="*50)
        return False
        
    if not MATLAB_OUTPUT_DIR.is_dir():
        print("="*50)
        print(f"ERROR: The path specified in 'MATLAB_OUTPUT_DIR' does not exist:")
        print(f"{MATLAB_OUTPUT_DIR}")
        print("Please check the path in main.py.")
        print("="*50)
        return False

    search_dir = MATLAB_OUTPUT_DIR
    pattern = "tmd_v7_simulation_*.json"
    
    try:
        simulation_files = list(search_dir.glob(pattern))
        
        if not simulation_files:
            print(f"Warning: No new simulation files found in '{search_dir}'. Using existing data.")
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
        print("Halting load: Could not update and no existing data file found.")
        return

    try:
        with open(DATA_FILE, 'r') as f:
            data = json.load(f)
            simulation_data = TMDSimulation(**data)
            print("✅ Simulation data loaded successfully.")
    except FileNotFoundError:
        print(f"Warning: {DATA_FILE} not found. API will return empty responses.")
        simulation_data = None
    except Exception as e:
        print(f"Error loading simulation data: {e}")
        simulation_data = None


@app.on_event("startup")
async def startup_event():
    """Load data on startup"""
    load_simulation_data()


# ============================================================================
# ORIGINAL API ENDPOINTS (Your existing endpoints)
# ============================================================================

@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "TMD Simulation API with Fuzzy Logic Control",
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
            "fuzzy_control": "/fuzzylogic",
            "fuzzy_batch": "/fuzzylogic-batch",
            "fuzzy_stats": "/fuzzy-stats",
            "fuzzy_history": "/fuzzy-history"
        }
    }

@app.get("/health", tags=["General"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "data_loaded": simulation_data is not None,
        "fuzzy_controller": "active",
        "computations": fuzzy_controller.computation_count
    }


@app.get("/simulation", response_model=TMDSimulation, tags=["Simulation"])
async def get_simulation():
    """Get complete simulation data"""
    load_simulation_data()
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


@app.get("/tmd-config", response_model=TMDConfiguration, tags=["TMD"])
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
    
    if start_time is None and end_time is None:
        return time_series
    
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
    """Reload simulation data from file"""
    load_simulation_data()
    return {
        "status": "success",
        "message": "Simulation data reloaded",
        "data_loaded": simulation_data is not None
    }


# ============================================================================
# FUZZY LOGIC CONTROL ENDPOINTS (New - Integrated from both controllers)
# ============================================================================

@app.post("/fuzzylogic", tags=["Fuzzy Control"])
async def fuzzy_logic_control(
    displacement: float = Query(..., description="Inter-story drift in meters"),
    velocity: float = Query(..., description="Inter-story drift velocity in m/s"),
    acceleration: Optional[float] = Query(None, description="Building acceleration in m/s² (optional)")
):
    """
    PRIMARY FUZZY LOGIC ENDPOINT - Use this shortcut to compute control force
    
    Receives real MATLAB data (displacement, velocity, acceleration) and
    returns the computed control force, automatically saving to JSON.
    
    Args:
        displacement: Inter-story drift in meters
        velocity: Inter-story drift velocity in m/s
        acceleration: Building acceleration in m/s² (optional, for logging)
    
    Returns:
        JSON with control force and metadata, saved to file
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
                "displacement_range_m": fuzzy_controller.displacement_range,
                "velocity_range_ms": fuzzy_controller.velocity_range,
                "force_range_kN": [fuzzy_controller.force_range[0]/1000, 
                                   fuzzy_controller.force_range[1]/1000]
            }
        }
        
        # Save to JSON file for MATLAB to read
        output_filename = f"fuzzy_output_{fuzzy_controller.computation_count:06d}.json"
        output_path = FUZZY_OUTPUT_DIR / output_filename
        
        with open(output_path, 'w') as f:
            json.dump(response, f, indent=2)
        
        # Also save as "latest" for easy MATLAB access
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
    Batch process multiple time steps from MATLAB simulation
    
    Processes entire time series and returns all control forces,
    saving complete results to JSON.
    
    Args:
        displacements: List of inter-story drifts in meters
        velocities: List of velocities in m/s
        accelerations: List of accelerations in m/s² (optional)
    
    Returns:
        JSON with all control forces and statistics
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
        
        # Prepare detailed response
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
    """
    Get recent fuzzy computation history
    
    Args:
        last_n: Number of recent computations to return (default 100)
    
    Returns:
        Recent computation history
    """
    history = fuzzy_controller.computation_history[-last_n:]
    
    return {
        "total_computations": fuzzy_controller.computation_count,
        "returned_count": len(history),
        "history": history
    }


@app.post("/fuzzy-save-history", tags=["Fuzzy Control"])
async def save_fuzzy_history():
    """
    Save complete fuzzy computation history to JSON file
    
    Returns:
        Path to saved file
    """
    try:
        filename = f"fuzzy_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = FUZZY_OUTPUT_DIR / filename
        
        saved_path = fuzzy_controller.save_computation_history(filepath)
        
        return {
            "status": "success",
            "message": "Computation history saved",
            "file_path": str(saved_path),
            "total_computations": fuzzy_controller.computation_count
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving history: {str(e)}")


# ============================================================================
# MATLAB INTEGRATION HELPERS
# ============================================================================

@app.get("/matlab-ready", tags=["MATLAB Integration"])
async def check_matlab_ready():
    """
    Check if API is ready for MATLAB integration
    Returns status of data loading and fuzzy controller
    """
    return {
        "api_status": "ready",
        "simulation_data_loaded": simulation_data is not None,
        "fuzzy_controller_active": True,
        "fuzzy_computations": fuzzy_controller.computation_count,
        "output_directory": str(FUZZY_OUTPUT_DIR),
        "matlab_instructions": {
            "single_computation": "POST /fuzzylogic?displacement=0.1&velocity=0.5&acceleration=2.0",
            "batch_computation": "POST /fuzzylogic-batch with arrays",
            "get_latest_result": f"Read {FUZZY_OUTPUT_DIR / 'fuzzy_output_latest.json'}"
        }
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*70)
    print("TMD SIMULATION API WITH COMPREHENSIVE FUZZY LOGIC CONTROL")
    print("="*70)
    print(f"Fuzzy Controller: Active")
    print(f"Force Range: ±{fuzzy_controller.force_range[1]/1000:.1f} kN")
    print(f"Output Directory: {FUZZY_OUTPUT_DIR}")
    print(f"\nPrimary Endpoint: POST /fuzzylogic")
    print(f"Port: 8001")
    print("="*70 + "\n")
    
    # Run on port 8001
    uvicorn.run(app, host="0.0.0.0", port=8001)

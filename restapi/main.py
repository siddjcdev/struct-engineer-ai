from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import List, Optional
import json
from pathlib import Path
from models import (
    TMDSimulation,
    BaselinePerformance,
    TMDResults,
    TMDConfiguration,
    Improvements,
    TimeSeriesData,
    PerformanceComparison,
    InputData
)

app = FastAPI(
    title="TMD Simulation API",
    description="REST API for Tuned Mass Damper (TMD) simulation data",
    version="1.0.0"
)

# Load simulation data
DATA_FILE = Path("data/simulation.json")
simulation_data: Optional[TMDSimulation] = None


def load_simulation_data():
    """Load simulation data from JSON file"""
    global simulation_data
    try:
        with open(DATA_FILE, 'r') as f:
            data = json.load(f)
            simulation_data = TMDSimulation(**data)
            print("Simulation data loaded successfully.")
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


@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "TMD Simulation API",
        "version": "1.0.0",
        "endpoints": {
            "simulation": "/simulation",
            "baseline": "/baseline",
            "tmd_results": "/tmd-results",
            "tmd_config": "/tmd-config",
            "improvements": "/improvements",
            "comparison": "/comparison",
            "time_series": "/time-series",
            "input": "/input"
        }
    }


@app.get("/health", tags=["General"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "data_loaded": simulation_data is not None
    }


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
    """Reload simulation data from file"""
    load_simulation_data()
    return {
        "status": "success",
        "message": "Simulation data reloaded",
        "data_loaded": simulation_data is not None
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
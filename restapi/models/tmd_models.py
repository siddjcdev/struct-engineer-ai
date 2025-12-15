from dataclasses import dataclass, field
from typing import List, Optional
# Note: The 'datetime' import was unused in the original models
# from datetime import datetime


@dataclass
class EarthquakeData:
    name: str
    magnitude: float
    dt: float
    duration: float
    n_points: int
    pga: float
    pga_g: float


@dataclass
class WindData:
    name: str
    mean_speed: float
    max_force_per_floor: float
    total_max_force: float


@dataclass
class InputData:
    earthquake: EarthquakeData
    use_wind: bool
    wind: WindData


@dataclass
class BaselinePerformance:
    # Pydantic's Field(...) with a description is converted to metadata
    DCR: float = field(metadata={"description": "Drift Concentration Ratio"})
    max_drift: float
    max_roof: float
    rms_roof: float
    rms_displacement: float
    rms_velocity: float
    rms_acceleration: float
    dcr_profile: List[float]


@dataclass
class TMDConfiguration:
    floor: int
    mass_ratio: float
    damping_ratio: float
    mass_kg: float
    stiffness: float
    damping: float
    natural_frequency: float
    optimization_score: float


@dataclass
class TMDResults:
    DCR: float
    max_drift: float
    max_roof: float
    rms_roof: float
    rms_displacement: float
    rms_velocity: float
    rms_acceleration: float
    dcr_profile: List[float]


@dataclass
class Improvements:
    dcr_reduction_pct: float
    drift_reduction_pct: float
    roof_reduction_pct: float
    rms_disp_reduction_pct: float
    rms_vel_reduction_pct: float
    rms_acc_reduction_pct: float


@dataclass
class V7Metadata:
    candidate_floors: List[int]
    performance_rating: str
    recommendation: str
    multi_objective_score: float


@dataclass
class TimeSeriesData:
    time: List[float]
    earthquake_acceleration: List[float]
    baseline_roof: List[float]
    tmd_roof: List[float]


@dataclass
class SimulationMetadata:
    n_floors: int
    time_step: float
    duration: float
    soft_story: int


@dataclass
class TMDSimulation:
    version: str
    timestamp: str
    metadata: SimulationMetadata
    input: InputData
    baseline: BaselinePerformance
    tmd: TMDConfiguration
    tmd_results: TMDResults
    improvements: Improvements
    v7: V7Metadata
    time_series: TimeSeriesData


@dataclass
class PerformanceComparison:
    metric: str
    baseline: float
    with_tmd: float
    improvement_pct: float
    unit: str

#Fuzzy Logic Model Starts Here

# Define the input data model for the API request
@dataclass
class BuildingState:
    # This matches the input from your flowchart
    current_drift: float
    current_velocity: float

# Define the output data model for the API response
@dataclass
class ControlOutput:
    # This matches the output from your flowchart
    control_force_newtons: float

#Fuzzy Logic Model Ends Here
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class EarthquakeData(BaseModel):
    name: str
    magnitude: float
    dt: float
    duration: float
    n_points: int
    pga: float
    pga_g: float


class WindData(BaseModel):
    name: str
    mean_speed: float
    max_force_per_floor: float
    total_max_force: float


class InputData(BaseModel):
    earthquake: EarthquakeData
    use_wind: bool
    wind: WindData


class BaselinePerformance(BaseModel):
    DCR: float = Field(..., description="Demand-to-Capacity Ratio")
    max_drift: float
    max_roof: float
    rms_roof: float
    rms_displacement: float
    rms_velocity: float
    rms_acceleration: float
    dcr_profile: List[float]


class TMDConfiguration(BaseModel):
    floor: int
    mass_ratio: float
    damping_ratio: float
    mass_kg: float
    stiffness: float
    damping: float
    natural_frequency: float
    optimization_score: float


class TMDResults(BaseModel):
    DCR: float
    max_drift: float
    max_roof: float
    rms_roof: float
    rms_displacement: float
    rms_velocity: float
    rms_acceleration: float
    dcr_profile: List[float]


class Improvements(BaseModel):
    dcr_reduction_pct: float
    drift_reduction_pct: float
    roof_reduction_pct: float
    rms_disp_reduction_pct: float
    rms_vel_reduction_pct: float
    rms_acc_reduction_pct: float


class V7Metadata(BaseModel):
    candidate_floors: List[int]
    performance_rating: str
    recommendation: str
    multi_objective_score: float


class TimeSeriesData(BaseModel):
    time: List[float]
    earthquake_acceleration: List[float]
    baseline_roof: List[float]
    tmd_roof: List[float]


class SimulationMetadata(BaseModel):
    n_floors: int
    time_step: float
    duration: float
    soft_story: int


class TMDSimulation(BaseModel):
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


class PerformanceComparison(BaseModel):
    metric: str
    baseline: float
    with_tmd: float
    improvement_pct: float
    unit: str
# TMD Simulation REST API

A FastAPI-based REST API for exposing Tuned Mass Damper (TMD) simulation data.

## Features

- 🚀 Fast and modern API built with FastAPI
- 📊 Complete simulation data access
- 🔍 Detailed performance metrics comparison
- 📈 Time series data with filtering
- 📝 Automatic API documentation (Swagger/OpenAPI)
- ✅ Data validation with Pydantic

## Project Structure

```
tmd-api/
├── main.py              # FastAPI application with endpoints
├── models.py            # Pydantic data models
├── requirements.txt     # Python dependencies
├── data/
│   └── simulation.json  # TMD simulation data
└── README.md           # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip

### Setup

1. Clone or download this project

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure your simulation data is in `data/simulation.json`

## Running the API

Start the server:
```bash
python main.py
```

Or use uvicorn directly:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at: `http://localhost:8000`

## API Endpoints

### General

- `GET /` - API information and endpoint list
- `GET /health` - Health check
- `POST /reload` - Reload simulation data from file

### Simulation Data

- `GET /simulation` - Get complete simulation data
- `GET /simulation/metadata` - Get simulation metadata only

### Performance Metrics

- `GET /baseline` - Get baseline performance (without TMD)
- `GET /tmd-results` - Get TMD performance results
- `GET /improvements` - Get performance improvements
- `GET /comparison` - Get side-by-side comparison
- `GET /dcr-profile` - Get DCR profile by floor

### TMD Configuration

- `GET /tmd-config` - Get TMD configuration parameters

### Input Data

- `GET /input` - Get earthquake and wind input parameters

### Time Series

- `GET /time-series` - Get time series data
  - Query params: `start_time`, `end_time` (optional)
- `GET /time-series/summary` - Get time series summary statistics

## Interactive Documentation

Once the server is running, visit:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Example Usage

### Using curl

Get baseline performance:
```bash
curl http://localhost:8000/baseline
```

Get comparison data:
```bash
curl http://localhost:8000/comparison
```

Get filtered time series (10s to 20s):
```bash
curl "http://localhost:8000/time-series?start_time=10&end_time=20"
```

### Using Python

```python
import requests

# Get complete simulation
response = requests.get("http://localhost:8000/simulation")
data = response.json()

# Get comparison
response = requests.get("http://localhost:8000/comparison")
comparison = response.json()

for metric in comparison:
    print(f"{metric['metric']}: {metric['improvement_pct']:.2f}% improvement")
```

### Using JavaScript/fetch

```javascript
// Get TMD results
fetch('http://localhost:8000/tmd-results')
  .then(response => response.json())
  .then(data => console.log(data));

// Get time series summary
fetch('http://localhost:8000/time-series/summary')
  .then(response => response.json())
  .then(summary => console.log(summary));
```

## Data Models

### Key Performance Metrics

- **DCR** (Demand-to-Capacity Ratio): Structural safety indicator
- **Max Drift**: Maximum inter-story drift
- **Max Roof**: Maximum roof displacement
- **RMS values**: Root-mean-square metrics for displacement, velocity, acceleration

### TMD Configuration

- Floor location
- Mass ratio (TMD mass / building mass)
- Damping ratio
- Natural frequency
- Optimization score

## Response Examples

### Baseline Performance
```json
{
  "DCR": 1.6728,
  "max_drift": 0.0566,
  "max_roof": 0.2352,
  "rms_roof": 0.0481,
  "rms_displacement": 0.0349,
  "rms_velocity": 0.0705,
  "rms_acceleration": 0.6166,
  "dcr_profile": [0.0369, 0.0342, ...]
}
```

### Comparison
```json
[
  {
    "metric": "DCR",
    "baseline": 1.6728,
    "with_tmd": 1.3721,
    "improvement_pct": 17.98,
    "unit": "ratio"
  },
  ...
]
```

## Error Handling

The API returns appropriate HTTP status codes:

- `200 OK` - Successful request
- `404 Not Found` - Data not found
- `422 Unprocessable Entity` - Invalid query parameters

## Development

### Adding New Endpoints

1. Define data models in `models.py`
2. Add endpoint function in `main.py`
3. Use type hints and Pydantic models for validation
4. Add appropriate tags for documentation organization

### Updating Simulation Data

Simply replace `data/simulation.json` with new data and call the `/reload` endpoint, or restart the server.

## License

This project is provided as-is for TMD simulation data analysis.

## Support

For issues or questions, please refer to the API documentation at `/docs` when the server is running.
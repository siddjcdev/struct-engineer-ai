# TMD Simulation REST API - Project Summary

## 🎯 Overview

This project provides a complete REST API for accessing Tuned Mass Damper (TMD) simulation data. Built with FastAPI, it offers fast, well-documented endpoints for analyzing building performance with and without TMD systems.

## 📁 Project Files

```
tmd-api/
├── main.py              # FastAPI application (400+ lines)
├── models.py            # Pydantic data models
├── requirements.txt     # Python dependencies
├── test_api.py         # API test/example script
├── Dockerfile          # Container configuration
├── docker-compose.yml  # Docker Compose setup
├── start.sh           # Quick start script
├── .gitignore         # Git ignore rules
├── README.md          # Full documentation
├── data/
│   └── simulation.json # TMD simulation data
└── SUMMARY.md         # This file
```

## 🚀 Quick Start

### Option 1: Direct Python

```bash
cd tmd-api
pip install -r requirements.txt
python main.py
```

### Option 2: Using Start Script

```bash
cd tmd-api
./start.sh
```

### Option 3: Docker

```bash
cd tmd-api
docker-compose up
```

## 📊 Key Endpoints

| Endpoint | Description | Example |
|----------|-------------|---------|
| `GET /` | API info | - |
| `GET /health` | Health check | - |
| `GET /simulation` | Complete data | Full simulation |
| `GET /baseline` | Without TMD | Performance metrics |
| `GET /tmd-results` | With TMD | Performance metrics |
| `GET /comparison` | Side-by-side | Baseline vs TMD |
| `GET /improvements` | % improvements | All reductions |
| `GET /tmd-config` | TMD parameters | Configuration |
| `GET /dcr-profile` | By floor | DCR per floor |
| `GET /time-series` | Time data | ?start_time=10&end_time=20 |
| `GET /input` | Inputs | Earthquake & wind |

## 🧪 Testing

Run the test script to verify all endpoints:

```bash
python test_api.py
```

Expected output:
- ✓ Health check passed
- ✓ All endpoints return valid data
- ✓ Performance metrics match
- ✓ Time series filtering works

## 📈 Example Results

From the provided simulation data:

**Performance Improvements:**
- DCR Reduction: 17.98%
- Drift Reduction: 21.31%
- Roof Displacement: 7.57%

**TMD Configuration:**
- Floor: 4
- Mass: 60,000 kg
- Mass Ratio: 0.3
- Damping Ratio: 0.03
- Natural Frequency: 3.56 Hz

**Seismic Input:**
- Event: Imperial Valley 1940 (El Centro)
- Magnitude: 6.9
- PGA: 0.346g

## 🔧 Customization

### Adding New Endpoints

1. Define model in `models.py`:
```python
class MyModel(BaseModel):
    field1: str
    field2: float
```

2. Add endpoint in `main.py`:
```python
@app.get("/my-endpoint", response_model=MyModel)
async def get_my_data():
    return MyModel(field1="value", field2=1.23)
```

### Updating Data

Replace `data/simulation.json` and call:
```bash
curl -X POST http://localhost:8000/reload
```

## 📚 Documentation

Once running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

Both provide:
- Interactive API testing
- Request/response schemas
- Example values
- Authentication info (if needed)

## 🔍 API Response Examples

### Comparison Endpoint
```json
[
  {
    "metric": "DCR",
    "baseline": 1.6728,
    "with_tmd": 1.3721,
    "improvement_pct": 17.98,
    "unit": "ratio"
  }
]
```

### Time Series Summary
```json
{
  "total_points": 12000,
  "duration": 119.99,
  "time_step": 0.01,
  "earthquake": {
    "max_acceleration": 3.3967,
    "min_acceleration": -2.9749
  }
}
```

## 🛠️ Technology Stack

- **FastAPI**: Modern, fast web framework
- **Pydantic**: Data validation using Python type hints
- **Uvicorn**: ASGI server
- **Python 3.8+**: Programming language

## 📦 Dependencies

- fastapi==0.104.1
- uvicorn==0.24.0
- pydantic==2.5.0
- python-multipart==0.0.6

## 🔒 Production Considerations

For production deployment:

1. **Security**: Add authentication (JWT, API keys)
2. **Rate Limiting**: Prevent abuse
3. **CORS**: Configure allowed origins
4. **Logging**: Add structured logging
5. **Monitoring**: Add health metrics
6. **Database**: Consider database for multiple simulations
7. **Caching**: Cache frequent queries
8. **HTTPS**: Use SSL/TLS

Example CORS setup:
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## 🧩 Integration Examples

### Python Client
```python
import requests

api = "http://localhost:8000"
comparison = requests.get(f"{api}/comparison").json()
```

### JavaScript/React
```javascript
const response = await fetch('http://localhost:8000/tmd-results');
const data = await response.json();
```

### cURL
```bash
curl -X GET "http://localhost:8000/time-series?start_time=10&end_time=20"
```

## 📞 Support

- Check `/docs` for interactive documentation
- Review `README.md` for detailed information
- Run `test_api.py` for working examples
- Check logs for error messages

## 🎓 Key Concepts

**TMD (Tuned Mass Damper)**: A passive vibration control device that reduces building motion during earthquakes or wind events.

**DCR (Demand-to-Capacity Ratio)**: Measures structural safety; values < 1.0 indicate safe design.

**RMS (Root Mean Square)**: Statistical measure of the magnitude of a varying quantity.

**Soft Story**: A floor with significantly reduced stiffness compared to adjacent floors.

## ✅ Checklist for Deployment

- [ ] Python 3.8+ installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] `data/simulation.json` present and valid
- [ ] Port 8000 available
- [ ] Firewall configured (if needed)
- [ ] `/health` endpoint returns healthy status
- [ ] Test script passes all checks
- [ ] Documentation accessible at `/docs`

## 🎉 Success!

Your TMD Simulation API is now ready to serve data to your applications, dashboards, or analysis tools!

Visit http://localhost:8000/docs to explore all available endpoints.
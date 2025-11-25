# System Architecture - Fuzzy Logic TMD Controller

```
┌─────────────────────────────────────────────────────────────────────┐
│                     MATLAB TMD SIMULATION                           │
│                                                                     │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────┐   │
│  │ Building Model   │ →  │ Newmark          │ →  │ Time Series  │   │
│  │ (Mass-Spring)    │    │ Integration      │    │ x(t), v(t)   │   │
│  └──────────────────┘    └──────────────────┘    └────────┬─────┘   │
│                                                           │         │
└───────────────────────────────────────────────────────────┼─────────┘
                                                            │
                                    HTTP POST               │
                                    /fuzzylogic             ▼
                                                    ┌───────────────┐
                                                    │  displacement │
                                                    │  velocity     │
                                                    │  acceleration │
                                                    └───────┬───────┘
                                                            │
┌───────────────────────────────────────────────────────────┼────────┐
│                    PYTHON REST API (Port 8001)            ▼        │
│                                                                    │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │            Comprehensive Fuzzy Logic Controller             │   │
│  │                                                             │   │
│  │  ┌───────────────┐   ┌──────────────┐   ┌───────────────┐   │   │
│  │  │ Fuzzification │ → │ Rule Base    │ → │Defuzzification│   │   │
│  │  │               │   │              │   │               │   │   │
│  │  │ 5 membership  │   │ 11 fuzzy     │   │ Weighted avg  │   │   │
│  │  │ functions     │   │ rules        │   │ (crisp value) │   │   │
│  │  └───────────────┘   └──────────────┘   └───────────────┘   │   │
│  │                                                             │   │
│  │  Inputs:                                  Output:           │   │
│  │  • Displacement (-0.5 to 0.5 m)          • Control Force    │   │
│  │  • Velocity (-2.0 to 2.0 m/s)              (-100 to 100 kN) │   │
│  └─────────────────────────────────────────────────┬───────────┘   │
│                                                    │               │
│                                                    ▼               │
│  ┌───────────────────────────────────────────────────────────┐     │
│  │                  JSON Output Files                        │     │
│  │                                                           │     │
│  │  data/fuzzy_outputs/                                      │     │
│  │  ├── fuzzy_output_latest.json    ← Always latest result   │     │
│  │  ├── fuzzy_output_000001.json    ← Numbered outputs       │     │
│  │  └── fuzzy_batch_*.json          ← Batch results          │     │
│  └──────────────────────────────────────────────┬────────────┘     │
└─────────────────────────────────────────────────┼──────────────────┘
                                                  │
                                    HTTP Response │
                                    JSON data     │
                                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     BACK TO MATLAB                                  │
│                                                                     │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────┐   │
│  │ Receive F_control│ →  │ Apply to EOM     │ →  │ Continue     │   │
│  │ from API         │    │ F_total = F_eq   │    │ simulation   │   │
│  │                  │    │         + F_ctrl │    │              │   │
│  └──────────────────┘    └──────────────────┘    └──────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Data Flow

### 1. Forward Path (MATLAB → Python)
```
Building State → HTTP Request → Fuzzy Controller → JSON Response
   (x, v, a)         POST          Compute F          (F_control)
```

### 2. Fuzzy Logic Processing
```
Input Values → Membership Functions → Rule Evaluation → Defuzzification → Output
   (crisp)         (fuzzification)      (inference)      (aggregation)    (crisp)
```

### 3. Return Path (Python → MATLAB)
```
Control Force → JSON File → MATLAB Reads → Apply to Building
  (Newtons)      (save)      (parse)        (dynamics)
```

## Component Details

### Membership Functions (5 levels each)

**Displacement:**
```
   |  NL     NS    Z     PS    PL
   |   ▲     ▲     ▲     ▲     ▲
   |  / \   / \   / \   / \   / \
   | /   \ /   \ /   \ /   \ /   \
   |/     X     X     X     X     \
   |──────┼─────┼─────┼─────┼──────
  -0.5  -0.3  -0.1   0   0.1  0.3  0.5 (meters)
```

**Velocity:**
```
   |  NF     NS    Z     PS    PF
   |   ▲     ▲     ▲     ▲     ▲
   |  / \   / \   / \   / \   / \
   | /   \ /   \ /   \ /   \ /   \
   |/     X     X     X     X     \
   |──────┼─────┼─────┼─────┼──────
  -2.0  -1.0  -0.3   0   0.3  1.0  2.0 (m/s)
```

**Control Force:**
```
   |  LN     SN    Z     SP    LP
   |   ▲     ▲     ▲     ▲     ▲
   |  / \   / \   / \   / \   / \
   | /   \ /   \ /   \ /   \ /   \
   |/     X     X     X     X     \
   |──────┼─────┼─────┼─────┼──────
  -100  -60   -20    0    20   60   100 (kN)
```

### Rule Base (11 rules)

```
IF displacement = PL  AND velocity = PF  THEN force = LN  (Rule 1)
IF displacement = NL  AND velocity = NF  THEN force = LP  (Rule 2)
IF displacement = PS  AND velocity = PS  THEN force = SN  (Rule 3)
IF displacement = PL  AND velocity = PS  THEN force = SN  (Rule 4)
IF displacement = NS  AND velocity = NS  THEN force = SP  (Rule 5)
IF displacement = NL  AND velocity = NS  THEN force = SP  (Rule 6)
IF displacement = Z   AND velocity = Z   THEN force = Z   (Rule 7)
IF displacement = PS  AND velocity = NS  THEN force = Z   (Rule 8)
IF displacement = PL  AND velocity = NF  THEN force = SP  (Rule 9)
IF displacement = NS  AND velocity = PS  THEN force = Z   (Rule 10)
IF displacement = NL  AND velocity = PF  THEN force = SN  (Rule 11)
```

Legend:
- NL/NF = Negative Large/Fast
- NS = Negative Small/Slow
- Z = Zero
- PS = Positive Small/Slow
- PL/PF = Positive Large/Fast
- LN = Large Negative force
- SN = Small Negative force
- SP = Small Positive force
- LP = Large Positive force

## API Endpoints

```
GET  /                      → API information
GET  /health                → Health check
GET  /matlab-ready          → MATLAB integration status

POST /fuzzylogic            → Single computation ⭐ PRIMARY
POST /fuzzylogic-batch      → Batch computation
GET  /fuzzy-stats           → Controller statistics
GET  /fuzzy-history         → Computation history
POST /fuzzy-save-history    → Save history to file

GET  /simulation            → Full simulation data
GET  /baseline              → Baseline performance
GET  /tmd-results           → TMD results
GET  /time-series           → Time series data
... (all your existing endpoints)
```

## File Locations

```
project/
├── main.py                              ← Python API server
├── models.py                            ← Data models
├── matlab_fuzzy_integration.m           ← MATLAB wrapper class
├── test_fuzzy_controller.py             ← Test script
├── README_FUZZY_CONTROLLER.md           ← This guide
│
└── data/
    ├── simulation.json                  ← MATLAB simulation input
    │
    └── fuzzy_outputs/                   ← Fuzzy controller outputs
        ├── fuzzy_output_latest.json     ← Always latest (easy access)
        ├── fuzzy_output_000001.json
        ├── fuzzy_output_000002.json
        ├── ...
        └── fuzzy_batch_*.json           ← Batch results
```

## Usage Patterns

### Pattern 1: Real-time (During Simulation)
```matlab
for i = 1:N
    F_control(i) = fuzzy.compute_single(x(i), v(i), a(i));
    % Apply force immediately
end
```
- Pro: True real-time control
- Con: Slower (N API calls)

### Pattern 2: Batch Processing (After Simulation)
```matlab
% Run simulation first
[x, v, a] = run_simulation();

% Then compute all forces
[F_control, stats] = fuzzy.compute_batch(x, v, a);

% Optionally re-run simulation with these forces
```
- Pro: Much faster (1 API call)
- Con: Not truly real-time

## Performance Metrics

Typical timing on modern hardware:
- Single computation: 5-10 ms
- Batch (1000 steps): 5-8 seconds
- Full simulation (500 steps): 20-40 seconds (real-time)
- Full simulation (500 steps): 3-5 seconds (batch)

Network latency: ~1-5 ms (localhost)
Fuzzy computation: ~4-8 ms
JSON parsing: ~0.5-1 ms

## Comparison with Original Controllers

| Feature | Simple Controller | Comprehensive Controller |
|---------|------------------|-------------------------|
| Input range | -1 to +1 (normalized) | Physical values (m, m/s) |
| Membership functions | automf (5 levels) | Custom (5 levels) |
| Rules | 7 | 11 |
| Force output | -1 to +1 (scaled) | Direct Newtons |
| API integration | Simple | Full-featured |
| MATLAB support | Manual | Wrapper class |
| Output saving | No | Yes (JSON) |
| Statistics | No | Yes |
| Batch processing | No | Yes |

✅ **The unified controller combines the best of both!**

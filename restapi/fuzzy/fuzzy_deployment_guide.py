"""
DEPLOYMENT GUIDE - FIXED FUZZY CONTROLLER
=========================================

Step-by-step instructions to deploy your fixed fuzzy controller

Author: Siddharth
Date: December 2025
"""

# ============================================================================
# STEP 1: LOCAL TESTING (DO THIS FIRST!)
# ============================================================================

"""
Before deploying to Cloud Run, test locally to make sure everything works.

1. Install dependencies:
   pip install numpy scikit-fuzzy matplotlib

2. Run the test suite:
   python test_fuzzy_local.py

3. Verify all tests pass:
   ✅ Basic functionality
   ✅ Force magnitude range
   ✅ Velocity opposition
   ✅ Batch processing
   ✅ Control surface visualization

Expected output:
- All forces should be within ±100 kN
- Forces should ALWAYS oppose velocity
- Control surface should look smooth
"""

# ============================================================================
# STEP 2: UPDATE YOUR FASTAPI APPLICATION
# ============================================================================

"""
Add the fixed fuzzy controller to your existing FastAPI app.

Your project structure should look like:
    your-api/
    ├── main.py                      # Your FastAPI app
    ├── fixed_fuzzy_controller.py    # ← Add this (from outputs)
    ├── requirements.txt             # Update this
    ├── Dockerfile                   # Update this
    └── ... (other files)

2a. Copy fixed_fuzzy_controller.py to your API folder

2b. Update main.py - Add these imports at the top:
    
    from fixed_fuzzy_controller import FixedFuzzyTMDController
    
2c. Initialize controller at startup (add this after app = FastAPI()):
    
    # Initialize controllers
    fuzzy_controller = FixedFuzzyTMDController()
    print("✅ Fuzzy controller loaded")

2d. Add the fuzzy endpoint (copy from fastapi_fuzzy_endpoint.py):
    
    @app.post("/fuzzylogic-batch")
    async def fuzzy_batch_predict(request: FuzzyBatchRequest):
        # ... (see fastapi_fuzzy_endpoint.py)

2e. Update your health check to confirm fuzzy is loaded:
    
    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "fuzzy_controller": "loaded",
            "nn_model_loaded": True
        }
"""

# ============================================================================
# STEP 3: UPDATE DEPENDENCIES
# ============================================================================

"""
Update requirements.txt to include scikit-fuzzy:

    fastapi
    uvicorn
    pydantic
    numpy
    torch
    scikit-fuzzy  # ← Add this line
    
If you have other dependencies, keep them too.
"""

# ============================================================================
# STEP 4: UPDATE DOCKERFILE
# ============================================================================

"""
Your Dockerfile should install all dependencies.

Example Dockerfile:

    FROM python:3.10-slim
    
    WORKDIR /app
    
    # Copy requirements and install
    COPY requirements.txt .
    RUN pip install --no-cache-dir -r requirements.txt
    
    # Copy application files
    COPY main.py .
    COPY fixed_fuzzy_controller.py .
    COPY your_nn_model.pth .  # If you have NN model
    
    # Expose port
    EXPOSE 8080
    
    # Run
    CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
"""

# ============================================================================
# STEP 5: TEST LOCALLY WITH DOCKER (OPTIONAL BUT RECOMMENDED)
# ============================================================================

"""
Before deploying to Cloud Run, test with Docker locally:

1. Build the image:
   docker build -t tmd-api-test .

2. Run the container:
   docker run -p 8080:8080 tmd-api-test

3. Test the endpoint:
   curl -X POST http://localhost:8080/fuzzylogic-batch \
     -H "Content-Type: application/json" \
     -d '{"displacements": [0.15, -0.10], "velocities": [0.8, -0.5]}'

4. Expected response:
   {
     "forces": [-45.23, 38.67],  // Approximate values
     "force_unit": "kN",
     "num_predictions": 2,
     "inference_time_ms": 15.2
   }

5. Check forces make sense:
   - First: disp=0.15 (right), vel=0.8 (right) → force should be NEGATIVE (push left)
   - Second: disp=-0.10 (left), vel=-0.5 (left) → force should be POSITIVE (push right)
"""

# ============================================================================
# STEP 6: DEPLOY TO CLOUD RUN
# ============================================================================

"""
Deploy to Google Cloud Run:

1. Authenticate with gcloud:
   gcloud auth login

2. Set your project:
   gcloud config set project YOUR_PROJECT_ID

3. Build and deploy in one command:
   gcloud run deploy tmd-api \
     --source . \
     --region us-east1 \
     --platform managed \
     --allow-unauthenticated \
     --memory 2Gi \
     --timeout 300

4. Wait for deployment (usually 2-5 minutes)

5. You'll get a URL like:
   https://tmd-api-887344515766.us-east1.run.app
"""

# ============================================================================
# STEP 7: TEST DEPLOYED API
# ============================================================================

"""
Test your deployed API:

1. Health check:
   curl https://YOUR-URL.run.app/health

2. Fuzzy test endpoint:
   curl https://YOUR-URL.run.app/fuzzylogic/test

3. Batch prediction:
   curl -X POST https://YOUR-URL.run.app/fuzzylogic-batch \
     -H "Content-Type: application/json" \
     -d '{"displacements": [0.15, -0.10, 0.0], "velocities": [0.8, -0.5, 0.0]}'

4. Expected response:
   {
     "forces": [-45.23, 38.67, 0.12],  // Should oppose velocity!
     "force_unit": "kN",
     "num_predictions": 3,
     "inference_time_ms": 12.5
   }
"""

# ============================================================================
# STEP 8: TEST FROM MATLAB
# ============================================================================

"""
Test from MATLAB to make sure integration works:

MATLAB code:

    API_URL = 'https://YOUR-URL.run.app';
    
    % Test data (RELATIVE motion: TMD - roof)
    test_displacements = [0.15, -0.10, 0.0];
    test_velocities = [0.8, -0.5, 0.0];
    
    % Create request
    batch_data = struct('displacements', test_displacements, ...
                        'velocities', test_velocities);
    json_data = jsonencode(batch_data);
    
    % Call API
    options = weboptions('MediaType', 'application/json', ...
                         'ContentType', 'json', ...
                         'Timeout', 30);
    
    response = webwrite([API_URL '/fuzzylogic-batch'], json_data, options);
    
    % Display results
    fprintf('Forces (kN): ');
    disp(response.forces);
    
Expected MATLAB output:
    Forces (kN): -45.23  38.67  0.12
"""

# ============================================================================
# STEP 9: UPDATE BATCH COMPARISON SCRIPT
# ============================================================================

"""
Now update your batch comparison to use RELATIVE motion.

Key changes needed in batch_compare_peer_datasets.m:

OLD (WRONG):
    roof_disp = disp_passive(12, :);  // Absolute roof displacement
    roof_vel = gradient(roof_disp, dt);
    forces_fuzzy = get_control_forces_from_api(API_URL, 'fuzzy', roof_disp, roof_vel);

NEW (CORRECT):
    roof_disp = disp_passive(12, :);
    roof_vel = gradient(roof_disp, dt);
    tmd_disp = disp_passive(13, :);
    tmd_vel = gradient(tmd_disp, dt);
    
    % Use RELATIVE motion
    relative_disp = tmd_disp - roof_disp;
    relative_vel = tmd_vel - roof_vel;
    
    forces_fuzzy = get_control_forces_from_api(API_URL, 'fuzzy', relative_disp, relative_vel);

Also update simulate_building_with_tmd to use Newton's 3rd law:
    
    if is_active
        F_control = control_forces(i);
        F_control = max(min(F_control, params.max_force), -params.max_force);
        
        % Apply with Newton's 3rd law
        F_eq(end-1) = F_eq(end-1) - F_control;  // Roof
        F_eq(end) = F_eq(end) + F_control;      // TMD
    end
"""

# ============================================================================
# TROUBLESHOOTING
# ============================================================================

"""
Common issues and solutions:

ISSUE: "ModuleNotFoundError: No module named 'skfuzzy'"
FIX: Add 'scikit-fuzzy' to requirements.txt and rebuild

ISSUE: Forces are still making things worse
FIX: Check that you're passing RELATIVE motion (TMD - roof), not absolute

ISSUE: API returns 500 error
FIX: Check Cloud Run logs:
     gcloud logging read "resource.type=cloud_run_revision" --limit 50

ISSUE: Forces have wrong sign (making vibrations worse)
FIX: Verify fuzzy rules in fixed_fuzzy_controller.py
     Run test_fuzzy_local.py to check velocity opposition

ISSUE: MATLAB can't connect to API
FIX: Check API URL is correct
     Test with curl first to verify API works
     Check firewall/network settings
"""

# ============================================================================
# EXPECTED PERFORMANCE
# ============================================================================

"""
After deploying fixed fuzzy controller, you should see:

Passive TMD:       31.53 cm (baseline)
Fixed Fuzzy TMD:   ~26 cm (15-20% improvement) ✅

If you still see worse performance:
1. Double-check you're using RELATIVE motion
2. Verify Newton's 3rd law is applied in simulation
3. Run test_fuzzy_local.py to verify controller logic

The improvement should be consistent across scenarios:
- El Centro: 15-18%
- Small earthquake: 16-20%
- Large earthquake: 15-19%
"""

# ============================================================================
# CHECKLIST
# ============================================================================

"""
Before considering fuzzy controller "done", verify:

□ test_fuzzy_local.py passes all tests
□ Control surface visualization looks smooth
□ Forces ALWAYS oppose velocity (critical!)
□ API health check shows fuzzy_controller: loaded
□ Test endpoint /fuzzylogic/test returns correct sign
□ Batch endpoint works from curl
□ MATLAB can call API and get correct forces
□ Simulation uses RELATIVE motion (TMD - roof)
□ Simulation applies Newton's 3rd law
□ Performance improvement is 15-20% vs passive
□ Results are consistent across different earthquakes

Once all boxes are checked, fuzzy controller is READY! ✅
"""

print(__doc__)
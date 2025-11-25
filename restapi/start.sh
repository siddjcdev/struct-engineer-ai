#!/bin/bash

echo "================================"
echo "TMD Simulation API - Quick Start"
echo "================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

echo "✓ Python 3 found"

# Check if requirements are installed
if ! python3 -c "import fastapi" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

echo "✓ Dependencies installed"

# Check if data file exists
if [ ! -f "data/simulation.json" ]; then
    echo "Warning: data/simulation.json not found"
    echo "Please ensure your simulation data is in the data directory"
fi

echo "✓ Starting API server..."
echo ""
echo "The API will be available at: http://localhost:8000"
echo "Interactive docs at: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the server
python3 main.py
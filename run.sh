#!/bin/bash
# ==============================================================================
# ParkVision AI - One-Click Launcher for macOS
# ==============================================================================

PROJECT_DIR="/Users/abhishekkumar/Desktop/parking_space_detection"
cd "$PROJECT_DIR" || exit 1

# Detect Python interpreter
if [ -f "/opt/anaconda3/bin/python" ]; then
    PYTHON_CMD="/opt/anaconda3/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
else
    PYTHON_CMD="python"
fi

echo "============================================================"
echo "🅿️  STARTING PARKVISION AI - SMART PARKING DETECTION"
echo "============================================================"
echo "🐍 Python Interpreter: $PYTHON_CMD"
echo "📂 Project Directory : $PROJECT_DIR"
echo "============================================================"

# Check if model exists, if not train it
if [ ! -f "$PROJECT_DIR/model_final.pth" ]; then
    echo "⚡ Model not found. Training CNN on dataset..."
    $PYTHON_CMD "$PROJECT_DIR/train.py"
fi

# Check if sample video exists, if not generate it
if [ ! -f "$PROJECT_DIR/car_test.mp4" ]; then
    echo "🎬 Generating test video feed..."
    $PYTHON_CMD "$PROJECT_DIR/generate_sample_video.py"
fi

echo "🚀 Launching Flask Web Server at http://127.0.0.1:5000 ..."
$PYTHON_CMD "$PROJECT_DIR/app.py"

# 🅿️ ParkVision AI - Parking Space Detection using Flask, OpenCV & CNN

An intelligent real-time parking space detection and monitoring platform built for macOS using **Flask**, **OpenCV**, and **Convolutional Neural Networks (PyTorch with Apple Silicon MPS GPU Acceleration)**.

---

## 🌟 Key Features

- **Real-Time Video Stream**: Live video streaming with dynamic bounding box overlays (Green = Available, Red = Occupied) and confidence percentages.
- **Deep Learning CNN Model**: High-accuracy Convolutional Neural Network trained directly on your cropped parking spot dataset.
- **Live Interactive Dashboard**: Modern UI with real-time KPI counters (Total Capacity, Available Slots, Occupied Slots, Occupancy Rate %), real-time Chart.js telemetry, and per-slot badges.
- **Interactive Space Picker Tool**: Visual OpenCV GUI tool (`parking_space_picker.py`) to mark, adjust, or delete parking spot bounding boxes on any video feed or image.
- **REST API (`/space_count`)**: Real-time JSON endpoint for third-party integrations, mobile apps, or smart signage.
- **Multi-Source Support**: Switch seamlessly between test video (`car_test.mp4`), parking simulation, and MacBook FaceTime / USB webcam.

---

## 📂 Where the Dataset is Located & How it is Trained

Your dataset is stored on your MacBook at:
- **Training Set**: `/Users/abhishekkumar/Desktop/train_data/train/`
  - `empty/`: 98 images of vacant parking bays.
  - `occupied/`: 334 images of parking bays with vehicles.
- **Testing Set**: `/Users/abhishekkumar/Desktop/train_data/test/`
  - `empty/`: 38 images.
  - `occupied/`: 126 images.

### 🏋️ How to Train the Model
Run the automated training script:
```bash
python train.py
```
This script:
1. Loads and validates all images from your `train_data` directory.
2. Applies data augmentations (horizontal flip, random rotation, color jitter) to handle daylight and shadow variations.
3. Utilizes **Apple Silicon GPU acceleration (`mps`)** for ultra-fast training (~14 seconds).
4. Evaluates on the test split, outputs accuracy metrics & confusion matrix, and saves the weights to `model_final.pth`.

You can also explore and run the interactive **`training_notebook.ipynb`** in Jupyter Notebook or VS Code.

---

## 🚀 Quick Start Guide (MacBook)

### 1. One-Click Launch
Open your Terminal and run:
```bash
cd /Users/abhishekkumar/Desktop/parking_space_detection
./run.sh
```

### 2. Manual Run
```bash
cd /Users/abhishekkumar/Desktop/parking_space_detection

# 1. Train the model (if not already trained)
python train.py

# 2. Start the Flask application
python app.py
```
Then open your browser at: **`http://127.0.0.1:5000`**

---

## 🛠️ Interactive Parking Space Picker (Mark Custom Spots)

To mark parking spots on any custom camera view or image:
```bash
python parking_space_picker.py
```
**Controls**:
- **Left Click**: Place a parking spot bounding box.
- **Right Click**: Delete the bounding box under the cursor.
- **`s`**: Save positions to `carposition.pkl`.
- **`c`**: Clear all positions.
- **`q` / `ESC`**: Save and exit.

---

## 📡 REST API Documentation

### `GET /space_count`
Returns real-time occupancy statistics and individual slot statuses:
```json
{
  "status": "success",
  "timestamp": "2026-08-26 09:30:15",
  "total_spaces": 16,
  "free_spaces": 6,
  "occupied_spaces": 10,
  "occupancy_rate": 62.5,
  "fps": 28.4,
  "spaces": [
    {
      "id": 1,
      "is_occupied": true,
      "status": "OCCUPIED",
      "confidence": 99.8,
      "coordinates": [140, 150, 100, 60]
    },
    {
      "id": 2,
      "is_occupied": false,
      "status": "FREE",
      "confidence": 99.4,
      "coordinates": [265, 150, 100, 60]
    }
  ]
}
```

---

## 📁 Project Structure

```
parking_space_detection/
├── app.py                     # Main Flask web application & video streamer
├── train.py                   # Automated CNN model training script
├── generate_sample_video.py   # Generates test video & parking lot background
├── parking_space_picker.py    # GUI tool to mark parking spot boxes
├── training_notebook.ipynb    # Step-by-step Jupyter Notebook for dataset training
├── model_final.pth            # Trained CNN weights
├── carposition.pkl            # Pickled parking slot coordinates
├── car_test.mp4               # Test video feed
├── car_park_bg.jpg            # Reference parking lot background image
├── requirements.txt           # Python package dependencies
├── run.sh                     # One-click launch script
├── templates/
│   └── index.html             # Modern responsive web dashboard
├── static/
│   ├── css/style.css          # Custom styling & animations
│   └── js/main.js             # Real-time telemetry, Chart.js & API polling
└── models/
    └── training_evaluation.png # Loss/Accuracy curve & Confusion Matrix plot
```

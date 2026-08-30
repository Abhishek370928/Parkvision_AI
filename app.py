"""
Parking Space Detection Web Application
Built with Flask, OpenCV, PyTorch/CNN, and Real-time Video Streaming.
"""

import os
import sys
import time
import pickle
import threading
import socket
from datetime import datetime
from PIL import Image

from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

# Initialize Flask App
app = Flask(__name__)

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_final.pth")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(BASE_DIR, "models", "model_final.pth")

POSITIONS_PATH = os.path.join(BASE_DIR, "carposition.pkl")
DEFAULT_VIDEO_PATH = os.path.join(BASE_DIR, "car_test.mp4")

# Set device (Apple Silicon MPS / CPU)
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"🚀 Using PyTorch inference device: {device}")

# CNN Architecture
class ParkingCNN(nn.Module):
    def __init__(self):
        super(ParkingCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Image transform pipeline
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Global State
class ParkingSystemState:
    def __init__(self):
        self.model = None
        self.positions = []
        self.video_source = DEFAULT_VIDEO_PATH
        self.cap = None
        self.lock = threading.Lock()
        
        # Latest detected metrics
        self.total_spaces = 0
        self.occupied_spaces = 0
        self.free_spaces = 0
        self.occupancy_rate = 0.0
        self.space_details = []
        self.last_update = datetime.now()
        self.fps = 0.0
        
        self.load_model()
        self.load_positions()
        self.init_video_capture()

    def load_model(self):
        try:
            self.model = ParkingCNN().to(device)
            if os.path.exists(MODEL_PATH):
                self.model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
                self.model.eval()
                print(f" Loaded CNN Model from {MODEL_PATH}")
            else:
                print(f"⚠️ Model file not found at {MODEL_PATH}, please run train.py")
        except Exception as e:
            print(f"❌ Error loading model: {e}")

    def load_positions(self):
        try:
            if os.path.exists(POSITIONS_PATH):
                with open(POSITIONS_PATH, "rb") as f:
                    self.positions = pickle.load(f)
                print(f" Loaded {len(self.positions)} slot coordinates from {POSITIONS_PATH}")
            else:
                # Default 16 positions
                self.positions = []
                start_x, step_x = 140, 125
                for i in range(8):
                    self.positions.append((start_x + i * step_x, 150, 100, 60))
                for i in range(8):
                    self.positions.append((start_x + i * step_x, 450, 100, 60))
                with open(POSITIONS_PATH, "wb") as f:
                    pickle.dump(self.positions, f)
                print("⚠️ Created default positions in carposition.pkl")
        except Exception as e:
            print(f"❌ Error loading positions: {e}")

    def init_video_capture(self):
        with self.lock:
            if self.cap is not None:
                self.cap.release()
            
            # Handle webcam vs video file
            if str(self.video_source).isdigit():
                self.cap = cv2.VideoCapture(int(self.video_source))
            else:
                self.cap = cv2.VideoCapture(self.video_source)

            if not self.cap.isOpened():
                print(f"⚠️ Could not open video source: {self.video_source}")

    def switch_source(self, new_source):
        self.video_source = new_source
        self.init_video_capture()
        print(f"🔄 Switched video source to: {new_source}")

state = ParkingSystemState()

def process_frame(frame):
    """Crops each parking spot, batches inference with CNN, and draws overlays."""
    if frame is None or len(state.positions) == 0:
        return frame

    h_frame, w_frame = frame.shape[:2]
    batch_tensors = []
    valid_indices = []

    # 1. Crop and prepare batch
    for idx, (x, y, w, h) in enumerate(state.positions):
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w_frame, x + w), min(h_frame, y + h)

        if (x2 - x1) < 10 or (y2 - y1) < 10:
            continue

        crop = frame[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)
        tensor_img = transform(pil_img)
        batch_tensors.append(tensor_img)
        valid_indices.append((idx, x, y, w, h))

    if not batch_tensors or state.model is None:
        return frame

    # 2. Batch Inference
    input_batch = torch.stack(batch_tensors).to(device)
    with torch.no_grad():
        outputs = state.model(input_batch)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1).cpu().numpy()
        confidences = torch.max(probs, dim=1).values.cpu().numpy()

    # 3. Aggregate Stats & Draw
    free_count = 0
    occupied_count = 0
    details = []

    overlay = frame.copy()

    for i, (slot_idx, x, y, w, h) in enumerate(valid_indices):
        is_occupied = bool(preds[i] == 1)
        conf = float(confidences[i]) * 100.0

        if is_occupied:
            occupied_count += 1
            color = (0, 30, 220)       # Red
            fill_color = (0, 0, 180)
            status_text = "OCCUPIED"
        else:
            free_count += 1
            color = (0, 200, 30)       # Green
            fill_color = (0, 150, 20)
            status_text = "FREE"

        details.append({
            "id": slot_idx + 1,
            "is_occupied": is_occupied,
            "status": status_text,
            "confidence": round(conf, 1),
            "coordinates": [x, y, w, h]
        })

        # Draw semi-transparent fill
        cv2.rectangle(overlay, (x, y), (x + w, y + h), fill_color, -1)
        # Draw crisp border
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Draw badge label
        badge_text = f"P{slot_idx+1:02d}: {status_text} {conf:.0f}%"
        cv2.rectangle(frame, (x, y - 18), (x + w, y), color, -1)
        cv2.putText(frame, badge_text, (x + 3, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    # Blend semi-transparent slot fills
    alpha = 0.22
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Update Global State Metrics
    total = len(state.positions)
    occ_rate = round((occupied_count / total * 100.0), 1) if total > 0 else 0.0

    state.total_spaces = total
    state.occupied_spaces = occupied_count
    state.free_spaces = free_count
    state.occupancy_rate = occ_rate
    state.space_details = details
    state.last_update = datetime.now()

    # 4. Top Status Header HUD
    header_h = 50
    cv2.rectangle(frame, (0, 0), (w_frame, header_h), (20, 20, 25), -1)
    cv2.line(frame, (0, header_h), (w_frame, header_h), (60, 60, 70), 1)

    # Title
    cv2.putText(frame, "AI SMART PARKING DETECTOR", (20, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # Metrics Pills
    # Total
    cv2.putText(frame, f"TOTAL: {total}", (w_frame - 540, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)
    # Available (Green)
    cv2.putText(frame, f"FREE: {free_count}", (w_frame - 400, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 240, 60), 2, cv2.LINE_AA)
    # Occupied (Red)
    cv2.putText(frame, f"OCCUPIED: {occupied_count}", (w_frame - 270, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 80, 255), 2, cv2.LINE_AA)
    # Rate
    cv2.putText(frame, f"{occ_rate}% FULL", (w_frame - 110, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 255), 2, cv2.LINE_AA)

    return frame

def generate_video_stream():
    """Video streaming generator function."""
    prev_time = time.time()
    while True:
        with state.lock:
            if state.cap is None or not state.cap.isOpened():
                state.init_video_capture()
                time.sleep(0.1)
                continue

            ret, frame = state.cap.read()
            if not ret:
                # Loop video if it's a file
                if not str(state.video_source).isdigit():
                    state.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = state.cap.read()
                
                if not ret:
                    time.sleep(0.1)
                    continue

        curr_time = time.time()
        fps = 1.0 / max(curr_time - prev_time, 1e-4)
        prev_time = curr_time
        state.fps = round(fps, 1)

        # Process frame with CNN
        processed = process_frame(frame)

        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', processed, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # Frame pacing
        time.sleep(0.03)

# ----------------- Routes ----------------- #

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_video_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/space_count')
@app.route('/api/space_count')
def space_count():
    """REST API endpoint returning current parking occupancy statistics."""
    return jsonify({
        "status": "success",
        "timestamp": state.last_update.strftime("%Y-%m-%d %H:%M:%S"),
        "total_spaces": state.total_spaces,
        "free_spaces": state.free_spaces,
        "occupied_spaces": state.occupied_spaces,
        "occupancy_rate": state.occupancy_rate,
        "fps": state.fps,
        "spaces": state.space_details
    })

@app.route('/api/switch_source', methods=['POST'])
def switch_source():
    """Allows user to switch video source dynamically."""
    data = request.get_json() or {}
    source_type = data.get('source', 'video')

    if source_type == 'camera' or source_type == 'webcam':
        state.switch_source(0)
    elif source_type == 'video':
        state.switch_source(DEFAULT_VIDEO_PATH)
    elif source_type == 'bg':
        bg_path = os.path.join(BASE_DIR, "car_park_bg.jpg")
        state.switch_source(bg_path)
    else:
        return jsonify({"status": "error", "message": "Invalid source"}), 400

    return jsonify({"status": "success", "source": state.video_source})

@app.route('/api/reload_positions', methods=['POST'])
def reload_positions():
    """Reloads parking positions from pickle without server restart."""
    state.load_positions()
    return jsonify({"status": "success", "total_slots": len(state.positions)})

@app.route('/api/retrain', methods=['POST'])
def retrain():
    """Spawns training process in background thread."""
    def run_train():
        from train import train_model
        train_model()
        state.load_model()

    thread = threading.Thread(target=run_train)
    thread.daemon = True
    thread.start()
    return jsonify({"status": "success", "message": "Training started in background"})

def find_available_port(preferred_port=5000):
    for port in [preferred_port, 5001, 8000, 8080]:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('127.0.0.1', port)) != 0:
                return port
    return preferred_port

if __name__ == '__main__':
    port = int(os.environ.get('PORT', find_available_port(5000)))
    print(f"🅿️ Starting Parking Space Detection Flask Server at http://127.0.0.1:{port} ...")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)

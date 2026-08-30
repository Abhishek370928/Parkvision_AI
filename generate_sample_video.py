"""
Sample Video & Parking Lot Generator
Generates a realistic parking lot video (`car_test.mp4`), a reference image (`car_park_bg.jpg`),
and the corresponding `carposition.pkl` parking slot coordinate mapping using spot patches from the dataset.
"""

import os
import glob
import random
import pickle
import cv2
import numpy as np

def create_sample_parking_lot_and_video():
    project_dir = "/Users/abhishekkumar/Desktop/parking_space_detection"
    train_data_dir = "/Users/abhishekkumar/Desktop/train_data"
    
    empty_imgs = glob.glob(f"{train_data_dir}/**/empty/*.jpg", recursive=True) + \
                 glob.glob(f"{train_data_dir}/**/empty/*.png", recursive=True) + \
                 glob.glob("/Users/abhishekkumar/Desktop/empty/*.jpg")
                 
    occupied_imgs = glob.glob(f"{train_data_dir}/**/occupied/*.jpg", recursive=True) + \
                    glob.glob(f"{train_data_dir}/**/occupied/*.png", recursive=True) + \
                    glob.glob("/Users/abhishekkumar/Desktop/occupied/*.jpg")
                    
    empty_imgs = list(set(empty_imgs))
    occupied_imgs = list(set(occupied_imgs))

    print(f"Loaded {len(empty_imgs)} empty patches, {len(occupied_imgs)} occupied patches.")

    # Video & Canvas dimensions
    width, height = 1280, 720
    slot_w, slot_h = 100, 60  # Aspect ratio matching parking slots
    
    # Define 16 parking bay positions in 2 rows
    # Row 1 (8 spots top)
    # Row 2 (8 spots bottom)
    positions = []
    
    start_x = 140
    step_x = 125
    y_row1 = 150
    y_row2 = 450

    for i in range(8):
        positions.append((start_x + i * step_x, y_row1, slot_w, slot_h))
    for i in range(8):
        positions.append((start_x + i * step_x, y_row2, slot_w, slot_h))

    # Save positions to carposition.pkl
    pkl_path = os.path.join(project_dir, "carposition.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(positions, f)
    print(f"💾 Saved {len(positions)} parking positions to {pkl_path}")

    # Generate Parking Lot Background Image
    bg_img = np.zeros((height, width, 3), dtype=np.uint8)
    bg_img[:] = (55, 55, 60)  # Dark asphalt color

    # Draw Road Markings & Driving Lanes
    # Driving Lane in center
    cv2.rectangle(bg_img, (50, 260), (1230, 400), (45, 45, 48), -1)
    # Lane divider dash lines
    for x in range(80, 1200, 80):
        cv2.line(bg_img, (x, 330), (x + 40, 330), (255, 255, 255), 3)

    # Draw parking boundary white lines for each spot
    for idx, (x, y, w, h) in enumerate(positions):
        cv2.rectangle(bg_img, (x, y), (x + w, y + h), (200, 200, 200), 2)
        # Add slot label on asphalt
        cv2.putText(bg_img, f"P{idx+1:02d}", (x + 10, y - 10 if y == y_row1 else y + h + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2)

    # Parking lot header sign
    cv2.putText(bg_img, "AI SMART PARKING LOT - LIVE SURVEILLANCE FEED", (start_x, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    bg_path = os.path.join(project_dir, "car_park_bg.jpg")
    cv2.imwrite(bg_path, bg_img)
    print(f"📸 Saved parking background image to {bg_path}")

    # Generate 15-second simulation video (car_test.mp4) at 20 FPS (300 frames)
    # Cars enter and leave slots over time
    video_path = os.path.join(project_dir, "car_test.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 20
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    # Initial state of 16 slots: 10 occupied, 6 empty
    slot_states = [1 if random.random() < 0.65 else 0 for _ in range(16)]

    # Pre-cache resized image patches
    cached_empty = []
    for p in empty_imgs[:25]:
        img = cv2.imread(p)
        if img is not None:
            cached_empty.append(cv2.resize(img, (slot_w, slot_h)))
            
    cached_occupied = []
    for p in occupied_imgs[:50]:
        img = cv2.imread(p)
        if img is not None:
            cached_occupied.append(cv2.resize(img, (slot_w, slot_h)))

    if not cached_empty:
        cached_empty = [np.full((slot_h, slot_w, 3), (60, 60, 65), dtype=np.uint8)]
    if not cached_occupied:
        cached_occupied = [np.full((slot_h, slot_w, 3), (30, 30, 200), dtype=np.uint8)]

    # Assign persistent patch to each slot
    slot_patches = []
    for i in range(16):
        e_patch = random.choice(cached_empty)
        o_patch = random.choice(cached_occupied)
        slot_patches.append({'empty': e_patch, 'occupied': o_patch})

    total_frames = 300  # 15 seconds loop
    print(f"🎬 Rendering {total_frames} frames of dynamic parking video...")

    for frame_idx in range(total_frames):
        frame = bg_img.copy()

        # Randomly toggle some cars leaving or arriving every 30-50 frames
        if frame_idx % 40 == 0 and frame_idx > 0:
            toggle_idx = random.randint(0, 15)
            slot_states[toggle_idx] = 1 - slot_states[toggle_idx]

        # Draw spots
        for idx, (x, y, w, h) in enumerate(positions):
            state = slot_states[idx]
            patch = slot_patches[idx]['occupied'] if state == 1 else slot_patches[idx]['empty']
            frame[y:y+h, x:x+w] = patch

        # Timestamp overlay
        timestamp_str = f"2026-08-26 09:{frame_idx//fps:02d}:{(frame_idx%fps)*3:02d} CAM_01 [HD]"
        cv2.putText(frame, timestamp_str, (width - 380, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        out.write(frame)

    out.release()
    print(f"✅ Generated sample video: {video_path} ({os.path.getsize(video_path)/1024:.1f} KB)")

if __name__ == "__main__":
    create_sample_parking_lot_and_video()

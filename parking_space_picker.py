"""
Interactive Parking Space Picker Tool (OpenCV GUI)
Allows users to visually mark parking slot bounding boxes on a camera feed, video frame, or background image.
Controls:
  - Left Mouse Click: Add a parking slot at the cursor position
  - Right Mouse Click: Delete the parking slot under the cursor
  - Key 's': Save positions to carposition.pkl
  - Key 'c': Clear all positions
  - Key 'r': Reset / reload saved positions
  - Key 'q' or 'ESC': Save and exit
"""

import os
import sys
import pickle
import cv2
import numpy as np

# Slot dimensions
SLOT_WIDTH = 100
SLOT_HEIGHT = 60

POSITIONS_FILE = "/Users/abhishekkumar/Desktop/parking_space_detection/carposition.pkl"
BACKGROUND_IMAGE = "/Users/abhishekkumar/Desktop/parking_space_detection/car_park_bg.jpg"
TEST_VIDEO = "/Users/abhishekkumar/Desktop/parking_space_detection/car_test.mp4"

# Global list of positions [(x, y, w, h)]
posList = []

def load_positions():
    global posList
    if os.path.exists(POSITIONS_FILE):
        try:
            with open(POSITIONS_FILE, 'rb') as f:
                posList = pickle.load(f)
            print(f" Loaded {len(posList)} existing parking slot positions from {POSITIONS_FILE}")
        except Exception as e:
            print(f"⚠️ Could not load positions: {e}")
            posList = []
    else:
        posList = []

def save_positions():
    with open(POSITIONS_FILE, 'wb') as f:
        pickle.dump(posList, f)
    print(f"💾 Successfully saved {len(posList)} parking slot positions to {POSITIONS_FILE}")

def mouse_click(events, x, y, flags, params):
    global posList
    if events == cv2.EVENT_LBUTTONDOWN:
        # Add new bounding box centered or top-left
        posList.append((x - SLOT_WIDTH // 2, y - SLOT_HEIGHT // 2, SLOT_WIDTH, SLOT_HEIGHT))
        print(f"➕ Added slot #{len(posList)} at ({x}, {y})")

    elif events == cv2.EVENT_RBUTTONDOWN:
        # Delete if clicked inside an existing box
        for i, pos in enumerate(posList):
            x1, y1, w, h = pos
            if x1 <= x <= x1 + w and y1 <= y <= y1 + h:
                deleted = posList.pop(i)
                print(f"➖ Removed slot #{i+1} at {deleted}")
                break

def main():
    load_positions()

    # Get sample image from background or first frame of video
    img = None
    if os.path.exists(BACKGROUND_IMAGE):
        img = cv2.imread(BACKGROUND_IMAGE)
    elif os.path.exists(TEST_VIDEO):
        cap = cv2.VideoCapture(TEST_VIDEO)
        ret, frame = cap.read()
        if ret:
            img = frame
        cap.release()

    if img is None:
        # Fallback blank canvas
        img = np.full((720, 1280, 3), 40, dtype=np.uint8)
        cv2.putText(img, "Parking Space Picker - Click to add parking slots", (100, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    window_name = "Parking Space Picker - [Left Click: Add | Right Click: Delete | S: Save | Q: Quit]"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    cv2.setMouseCallback(window_name, mouse_click)

    print("\n" + "=" * 60)
    print("🅿️ PARKING SPACE PICKER TOOL")
    print("=" * 60)
    print("🖱️  Left Click   : Place a parking spot box")
    print("🖱️  Right Click  : Delete a parking spot box")
    print("⌨️  's'           : Save positions to carposition.pkl")
    print("⌨️  'c'           : Clear all slots")
    print("⌨️  'r'           : Reload saved positions")
    print("⌨️  'q' / ESC     : Save & Exit")
    print("=" * 60 + "\n")

    while True:
        display_img = img.copy()

        # Draw all current parking slots
        for idx, (x, y, w, h) in enumerate(posList):
            cv2.rectangle(display_img, (x, y), (x + w, y + h), (255, 0, 255), 2)
            cv2.putText(display_img, f"#{idx+1}", (x + 5, y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Draw status banner
        cv2.rectangle(display_img, (0, 0), (display_img.shape[1], 40), (20, 20, 20), -1)
        cv2.putText(display_img, f"Total Marked Slots: {len(posList)} | Press 'S' to Save | 'Q' to Quit",
                    (20, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        cv2.imshow(window_name, display_img)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('s'):
            save_positions()
        elif key == ord('c'):
            posList = []
            print("🗑️ Cleared all parking spots.")
        elif key == ord('r'):
            load_positions()
        elif key == ord('q') or key == 27:
            save_positions()
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

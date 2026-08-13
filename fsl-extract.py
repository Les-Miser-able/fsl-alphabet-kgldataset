"""
Convert the Kaggle FSL Alphabet image dataset into landmark-based
X.npy / y.npy files suitable for a Conv1D + LSTM model.

Uses the new MediaPipe Tasks API (HandLandmarker), NOT the deprecated
mp.solutions.hands API.

Expected input folder structure (typical Kaggle FSL dataset layout):

    fsl-dataset/
        A/
            img1.jpg
            img2.jpg
            ...
        B/
            ...
        ...

Each image is passed through HandLandmarker to extract 21 hand
landmarks (x, y, z). Images where no hand is detected are skipped.

Output:
    X.npy  -> shape (N, 21, 3)  float32
    y.npy  -> shape (N,)        string labels (letter names)

SETUP (run once before this script):

1. Install dependencies:
    pip install mediapipe opencv-python numpy --break-system-packages

2. Download the hand landmarker model file and place it next to this
   script (or update MODEL_PATH below):
    https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task

   Or via command line:
    wget -O hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
"""

import os
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ---------------- CONFIG ----------------
DATASET_ROOT = "data/Collated"          # change to your dataset folder path
MODEL_PATH = "hand_landmarker.task"   # path to downloaded model file
OUTPUT_X = "extracted_data/X.npy"
OUTPUT_Y = "extracted_data/y.npy"
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")
MIN_HAND_DETECTION_CONFIDENCE = 0.5
# -----------------------------------------


def create_detector():
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=MIN_HAND_DETECTION_CONFIDENCE,
        running_mode=mp_vision.RunningMode.IMAGE,  # static images, not video
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def extract_landmarks(image_path, detector):
    """Return a (21, 3) array of hand landmarks, or None if no hand found."""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return None

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

    result = detector.detect(mp_image)

    if not result.hand_landmarks:
        return None

    landmarks = result.hand_landmarks[0]  # first detected hand
    coords = np.array(
        [[lm.x, lm.y, lm.z] for lm in landmarks],
        dtype=np.float32
    )
    return coords


def main():
    data = []
    labels = []
    skipped = 0
    per_class_count = {}

    detector = create_detector()

    class_folders = sorted(
        d for d in os.listdir(DATASET_ROOT)
        if os.path.isdir(os.path.join(DATASET_ROOT, d))
    )

    for label in class_folders:
        folder_path = os.path.join(DATASET_ROOT, label)
        image_files = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith(VALID_EXTENSIONS)
        ]

        count_for_class = 0
        for fname in image_files:
            img_path = os.path.join(folder_path, fname)
            coords = extract_landmarks(img_path, detector)

            if coords is None:
                skipped += 1
                continue

            data.append(coords)
            labels.append(label)
            count_for_class += 1

        per_class_count[label] = count_for_class
        print(f"[{label}] kept {count_for_class}/{len(image_files)} images")

    detector.close()

    X = np.array(data, dtype=np.float32)   # shape (N, 21, 3)
    y = np.array(labels)

    np.save(OUTPUT_X, X)
    np.save(OUTPUT_Y, y)

    print("\n--- Conversion complete ---")
    print(f"Total kept:    {len(X)}")
    print(f"Total skipped: {skipped}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Saved to: {OUTPUT_X}, {OUTPUT_Y}")

    print("\nPer-class counts (check for imbalance):")
    for label, cnt in per_class_count.items():
        print(f"  {label}: {cnt}")


if __name__ == "__main__":
    main()
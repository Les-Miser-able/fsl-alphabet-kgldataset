"""
Real-time FSL alphabet recognition using webcam.

Pipeline per frame:
    webcam frame -> MediaPipe HandLandmarker -> 21x3 landmarks
    -> normalize -> trained Conv1D+LSTM model -> predicted letter
    -> draw landmarks + prediction on screen

Requirements:
    pip install mediapipe opencv-python numpy tensorflow

You need:
    1. hand_landmarker.task       (MediaPipe model, same as before)
    2. your trained model file    (e.g. fsl_model.keras or fsl_model.h5)
    3. the list of class labels in the SAME order used during training
       (e.g. saved as label_classes.npy via LabelEncoder, or hardcode
       the sorted list of folder names: A, B, C, ...)

Press 'q' to quit the camera window.
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from tensorflow import keras

# ---------------- CONFIG ----------------
MODEL_TASK_PATH = "hand_landmarker.task"   # MediaPipe hand landmark model
TRAINED_MODEL_PATH = "fsl_model.keras"     # your trained Conv1D+LSTM model
LABEL_CLASSES_PATH = "label_classes.npy"   # array of class names, index-aligned to model output
NORM_MEAN_PATH = "norm_mean.npy"           # saved during training
NORM_STD_PATH = "norm_std.npy"             # saved during training
CONFIDENCE_THRESHOLD = 0.6                 # only show prediction above this confidence
# -----------------------------------------

# Fixed hand skeleton topology (21 landmarks) - hardcoded so we don't
# depend on mp.solutions (deprecated) or any specific mp.tasks naming.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index finger
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle finger
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring finger
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky + palm
]


def create_landmarker():
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_TASK_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=mp_vision.RunningMode.VIDEO,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def landmarks_to_array(hand_landmarks):
    """Convert MediaPipe landmark list to (21, 3) numpy array."""
    return np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks],
        dtype=np.float32
    )


def draw_landmarks(frame, hand_landmarks, w, h):
    """Draw landmark points and connections on the frame."""
    points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

    for connection in HAND_CONNECTIONS:
        start_idx, end_idx = connection
        cv2.line(frame, points[start_idx], points[end_idx], (0, 255, 0), 2)

    for point in points:
        cv2.circle(frame, point, 4, (0, 0, 255), -1)


def main():
    landmarker = create_landmarker()
    model = keras.models.load_model(TRAINED_MODEL_PATH)
    class_names = np.load(LABEL_CLASSES_PATH, allow_pickle=True)
    norm_mean = np.squeeze(np.load(NORM_MEAN_PATH))   # shape (3,)
    norm_std = np.squeeze(np.load(NORM_STD_PATH))      # shape (3,)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    frame_timestamp_ms = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # mirror for natural interaction
        h, w, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        frame_timestamp_ms += 33  # approx 30 fps timestamp increment
        result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        if result.hand_landmarks:
            hand_landmarks = result.hand_landmarks[0]

            draw_landmarks(frame, hand_landmarks, w, h)

            coords_raw = landmarks_to_array(hand_landmarks)
            print("norm_mean shape:", norm_mean.shape, "norm_std shape:", norm_std.shape)
            print("coords_raw shape:", coords_raw.shape)     # should be (21, 3)

            coords = (coords_raw - norm_mean) / norm_std
            print("coords after normalize:", coords.shape)   # should be (21, 3)

            input_data = np.expand_dims(coords, axis=0)     # shape (1, 21, 3)

            predictions = model.predict(input_data, verbose=0)[0]
            best_idx = np.argmax(predictions)
            confidence = predictions[best_idx]
            predicted_label = class_names[best_idx]

            if confidence >= CONFIDENCE_THRESHOLD:
                text = f"{predicted_label} ({confidence*100:.1f}%)"
                color = (0, 255, 0)
            else:
                text = f"Unsure ({predicted_label}, {confidence*100:.1f}%)"
                color = (0, 165, 255)

            cv2.putText(frame, text, (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        else:
            cv2.putText(frame, "No hand detected", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        cv2.imshow("FSL Alphabet Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()
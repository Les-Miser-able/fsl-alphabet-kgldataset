"""Shared two-hand sequence preprocessing. No camera opens on import."""
from pathlib import Path
import time
import numpy as np

ROOT = Path(__file__).resolve().parent
FRAMES = 30
LANDMARKS = 42
FEATURES = 4  # x, y, z, hand-present
HAND_ORDER = ("Left", "Right")  # MediaPipe labels, independent of result order
FORMAT = "fsl-two-hands-v2"
NORMALIZATION = "shared-first-wrist-xy_shared-palm-scale_local-z_presence"
DYNAMIC = {"J", "Z"}


def normalize_sequence(sequence):
    """Use one XY reference/scale for BOTH hands, retaining relative position."""
    x = np.asarray(sequence, dtype=np.float32).copy()
    if x.ndim != 4 or x.shape[1:] != (2, 21, 3) or len(x) == 0:
        raise ValueError("Expected raw landmarks shaped (T,2,21,3); absent slots use NaN.")
    present = np.isfinite(x).all(axis=(2, 3))
    if not present.any():
        raise ValueError("No detected hands.")
    lengths = np.linalg.norm(x[:, :, 9, :2] - x[:, :, 0, :2], axis=-1)
    scale = float(np.median(lengths[present]))
    if not np.isfinite(scale) or scale < 1e-5:
        raise ValueError("Degenerate hand scale; record the gesture again.")
    frame, slot = np.argwhere(present)[0]
    anchor = x[frame, slot, 0, :2].copy()
    x[:, :, :, :2] -= anchor
    # MediaPipe Z is local to each wrist: never invent relative depth between hands.
    x[:, :, :, 2] -= x[:, :, 0:1, 2].copy()
    x /= scale
    x[~present] = 0
    mask = np.broadcast_to(present[:, :, None, None], (*x.shape[:-1], 1))
    return np.concatenate([x, mask.astype(np.float32)], axis=-1).reshape(-1, LANDMARKS, FEATURES)


def prepare_clip(coords, timestamps):
    """Resample each hand separately; never extrapolate an absent hand."""
    x = np.asarray(coords, dtype=np.float32)
    t = np.asarray(timestamps, dtype=np.float64)
    if x.ndim != 4 or x.shape[1:] != (2, 21, 3) or t.shape != (len(x),):
        raise ValueError("Clip needs aligned (T,2,21,3) coordinates and timestamps. Re-extract old one-hand data.")
    if len(t) < 8 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("Need at least 8 frames with strictly increasing timestamps.")
    if not 0.25 <= t[-1] - t[0] <= 10:
        raise ValueError("Record a complete gesture lasting 0.25 to 10 seconds.")
    present = np.isfinite(x).all(axis=(2, 3))
    any_hand = present.any(axis=1)
    if not any_hand[0] or not any_hand[-1] or any_hand.mean() < 0.9:
        raise ValueError("Both hands missing at a boundary or in over 10% of frames.")
    if np.max(np.diff(t[any_hand])) > .25:
        raise ValueError("Tracking gap exceeds 0.25 seconds; record again.")
    target = np.linspace(t[0], t[-1], FRAMES)
    sampled = np.full((FRAMES, 2, 21, 3), np.nan, dtype=np.float32)
    for slot in range(2):
        valid = present[:, slot]
        vt = t[valid]
        if not len(vt):
            continue  # This may legitimately be a one-handed sign.
        flat = x[valid, slot].reshape(-1, 63)
        # A target is supported only by a detection or a short internal gap.
        hi = np.searchsorted(vt, target, side="left").clip(0, len(vt)-1)
        lo = np.maximum(hi-1, 0)
        supported = ((target >= vt[0]) & (target <= vt[-1]) &
                     ((vt[hi]-vt[lo] <= .25) | np.isclose(target, vt[hi], atol=1e-8, rtol=0)))
        values = np.stack([np.interp(target, vt, flat[:, j]) for j in range(63)], axis=1)
        sampled[supported, slot] = values[supported].reshape(-1, 21, 3)
    return normalize_sequence(sampled)


def prepare_still(coords):
    return normalize_sequence(np.repeat(np.asarray(coords)[None], FRAMES, axis=0))


def detector(video=False):
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    path = ROOT / "hand_landmarker.task"
    if not path.exists():
        raise FileNotFoundError(f"Missing MediaPipe model: {path}")
    return vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(path)),
        num_hands=2, min_hand_detection_confidence=0.5,
        running_mode=vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE))


def assign_hands(result):
    """Map detector output into stable Left,Right slots; reject ambiguous labels."""
    coords = np.full((2, 21, 3), np.nan, dtype=np.float32)
    used = set()
    for landmarks, categories in zip(result.hand_landmarks, result.handedness):
        if not categories:
            continue
        category = categories[0]
        if category.category_name not in HAND_ORDER or category.score < .6:
            continue
        slot = HAND_ORDER.index(category.category_name)
        if slot in used:
            return np.full((2, 21, 3), np.nan, dtype=np.float32)
        used.add(slot)
        coords[slot] = [[p.x, p.y, p.z] for p in landmarks]
    return coords


def detect_frame(frame, landmarker, timestamp_ms=None):
    import cv2
    import mediapipe as mp
    image = mp.Image(image_format=mp.ImageFormat.SRGB,
                     data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    result = (landmarker.detect(image) if timestamp_ms is None
              else landmarker.detect_for_video(image, timestamp_ms))
    return assign_hands(result)

def read_video(path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0 or fps > 240:
            raise ValueError("Video needs valid frame-rate metadata.")
        coords, times = [], []
        with detector(video=True) as landmarker:
            index = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                t = index / fps
                if t > 10:
                    raise ValueError("Trim video to one complete gesture, at most 10 seconds.")
                points = detect_frame(frame, landmarker, round(t * 1000))
                coords.append(points)
                times.append(t)
                index += 1
        return prepare_clip(coords, times)
    finally:
        cap.release()


def live_clips(on_clip, camera=0):
    """User-delimited clips, shared by collection and recognition. Raw, unmirrored input."""
    import cv2
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError("Cannot open camera. Check camera permissions and device index.")
    recording = False
    coords, times = [], []
    message = "SPACE starts/stops a gesture; C cancels; Q quits"
    last_ms = -1
    try:
        with detector(video=True) as landmarker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("Camera disconnected; unfinished clip discarded.")
                now = time.monotonic()
                timestamp = max(last_ms + 1, int(now * 1000))
                last_ms = timestamp
                points = detect_frame(frame, landmarker, timestamp)
                if recording:
                    coords.append(points)
                    times.append(now)
                    if times[-1] - times[0] > 10:
                        recording = False
                        coords, times = [], []
                        message = "Clip too long; discarded. SPACE to try again."
                h, w = frame.shape[:2]
                for slot, hand_points in enumerate(points):
                    if not np.isfinite(hand_points).all():
                        continue
                    color = (0, 255, 0) if slot == 0 else (255, 180, 0)
                    for x, y, _ in hand_points:
                        cv2.circle(frame, (int(x*w), int(y*h)), 3, color, -1)
                    wrist = hand_points[0]
                    cv2.putText(frame, HAND_ORDER[slot], (int(wrist[0]*w), int(wrist[1]*h)),
                                cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1)
                status = f"RECORDING {len(coords)} frames - SPACE to finish" if recording else message
                cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
                cv2.putText(frame, "One or two hands | SPACE start/stop | C cancel | Q quit",
                            (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.imshow("FSL sequence capture (unmirrored)", frame)
                key = cv2.waitKey(1) & 0xff
                if key == ord("q"):
                    break
                if key == ord("c"):
                    recording = False
                    coords, times = [], []
                    message = "Cancelled. SPACE to start."
                if key == ord(" "):
                    if not recording:
                        coords, times = [], []
                        recording = True
                    else:
                        recording = False
                        try:
                            sequence = prepare_clip(coords, times)
                            message = on_clip(sequence, np.asarray(coords), np.asarray(times))
                        except ValueError as error:
                            message = str(error)
                        print(message)
                        coords, times = [], []
    finally:
        cap.release()
        cv2.destroyAllWindows()

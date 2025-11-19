"""
Main application that composes detection and depth modules.
This file provides a `main()` function and can be run as a script.

It deliberately re-uses helpers and constants from `car_alert_simple.py` where appropriate
so we do not duplicate configuration values.
"""

import time
import sys
import cv2
import numpy as np
import threading

# --- Configuration (define here, do not import from other files) ---
WEIGHTS = "yolov10m.pt"
CAMERA_INDEX = 3
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
AREA_FRAC_THRESHOLD = 0.3
DISPLAY_MAX_WIDTH = 1280
TARGET_WIDTH = 544
TARGET_HEIGHT = 960

# Depth module removed — using detection only

import os
import detection

# Thread-safe shared frame for external consumers (e.g. a web server)
_frame_lock = threading.Lock()
_latest_frame = None
# optional callback that will be called with each new frame: Callable[[numpy.ndarray], None]
_frame_callback = None

def get_latest_frame():
    """Return a copy of the latest visualization frame (BGR numpy array) or None."""
    with _frame_lock:
        if _latest_frame is None:
            return None
        return _latest_frame.copy()

def set_frame_callback(cb):
    """Set an optional callback function called with each new frame.

    cb should accept one argument: the frame as a BGR numpy array.
    """
    global _frame_callback
    _frame_callback = cb

def _publish_frame(frame):
    """Internal: store and optionally callback with a copy of frame."""
    global _latest_frame
    try:
        fcpy = frame.copy()
    except Exception:
        fcpy = frame
    with _frame_lock:
        _latest_frame = fcpy
    if _frame_callback is not None:
        try:
            _frame_callback(fcpy)
        except Exception:
            pass

# Local helpers to avoid importing from `car_alert_simple`
def init_model(weights_path: str):
    """Load YOLO model from weights_path."""
    try:
        from ultralytics import YOLO
    except Exception:
        print("Error: ultralytics YOLO not available. Please install ultralytics.")
        return None
    if not os.path.exists(weights_path):
        print(f"Warning: weights file '{weights_path}' not found. Model load may fail if path is incorrect.")
    print(f"Loading YOLO model from {weights_path} ...")
    try:
        model = YOLO(weights_path)
        print("YOLO model loaded.")
        return model
    except Exception as e:
        print("Failed to load YOLO model:", e)
        return None


def init_camera(index: int, target_w: int, target_h: int):
    """Open camera and attempt to set target resolution; return cap, disp_w, disp_h."""
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"Cannot open camera {index}")
        sys.exit(2)
    cam_w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    cam_h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    print(f"Camera native resolution before set: {cam_w0} x {cam_h0}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(target_w))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(target_h))
    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    print(f"Camera resolution after set attempt: {cam_w} x {cam_h}")
    display_scale = 1.0
    if cam_w > DISPLAY_MAX_WIDTH and DISPLAY_MAX_WIDTH > 0:
        display_scale = DISPLAY_MAX_WIDTH / float(cam_w)
    disp_w = max(1, int(cam_w * display_scale))
    disp_h = max(1, int(cam_h * display_scale))
    return cap, disp_w, disp_h


def transform_frame_to_target(frame, target_w: int, target_h: int):
    h, w = frame.shape[:2]
    if (w, h) == (target_w, target_h):
        return frame
    scale = max(target_w / w, target_h / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    x0 = max(0, (new_w - target_w) // 2)
    y0 = max(0, (new_h - target_h) // 2)
    cropped = resized[y0:y0 + target_h, x0:x0 + target_w]
    ch, cw = cropped.shape[:2]
    if ch != target_h or cw != target_w:
        out = np.zeros((target_h, target_w, 3), dtype=frame.dtype)
        out[0:ch, 0:cw] = cropped
        return out
    return cropped


def init_depth_model():
    """Simple depth model loader compatible with depth.init_depth(loader=...).

    This mirrors the behaviour of the prior script but is kept local to avoid
    importing the original file.
    """
    try:
        import torch
    except Exception as e:
        print("Torch not available, cannot load depth model:", e)
        return None, None, None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device for depth:", device)
    model_name = "MiDaS"
    try:
        print(f"Loading depth model '{model_name}' from torch.hub ...")
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        lname = model_name.lower()
        if "small" in lname:
            transform = midas_transforms.small_transform
        elif "dpt" in lname or "hybrid" in lname:
            transform = midas_transforms.dpt_transform
        else:
            transform = midas_transforms.default_transform

        midas = torch.hub.load("intel-isl/MiDaS", model_name).to(device)
        midas.eval()
        return midas, transform, device
    except Exception as e:
        print("Failed to load depth model from torch.hub:", e)
        return None, None, None


def main():
    if init_model is None:
        print("Error: model loader not available")
        sys.exit(1)

    model = init_model(WEIGHTS)
    cap, disp_w, disp_h = init_camera(CAMERA_INDEX, TARGET_WIDTH, TARGET_HEIGHT)

    # Minimal warmup: run one small predict to initialize model internals
    try:
        test_img = (255 * np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8))
        _ = model.predict(source=test_img, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)
    except Exception:
        pass

    midas_model = None
    midas_transform = None
    midas_device = None

    print("Main app started. Press q to quit.")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break

        # center crop/resize to target
        from car_alert_simple import transform_frame_to_target
        frame_t = transform_frame_to_target(frame, TARGET_WIDTH, TARGET_HEIGHT)

        # detection
        dets = detection.detect_frame(model, frame_t, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD)
        vis, alert = detection.draw_detections(frame_t, dets, area_frac_threshold=AREA_FRAC_THRESHOLD)

        # publish the visualisation for external consumers (web server, etc.)
        try:
            _publish_frame(vis)
        except Exception:
            pass

        # depth processing removed — detection-only mode

        # display detections
        cv2.imshow("CarAlert", cv2.resize(vis, (disp_w, disp_h), interpolation=cv2.INTER_AREA))

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q')):
            break
        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

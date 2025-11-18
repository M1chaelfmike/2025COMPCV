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

# --- Configuration (define here, do not import from other files) ---
WEIGHTS = "yolov10m.pt"
CAMERA_INDEX = 3
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
AREA_FRAC_THRESHOLD = 0.3
DISPLAY_MAX_WIDTH = 1280
TARGET_WIDTH = 544
TARGET_HEIGHT = 960

# Depth options
ENABLE_DEPTH = True
DEPTH_EVERY_N_FRAMES = 3

from car_alert_simple import init_model, init_camera
# import only the depth loader function (we pass it into our depth module)
from car_alert_simple import init_depth_model
import detection
import depth


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
    if ENABLE_DEPTH:
        # pass the loader from car_alert_simple into our depth module
        midas_model, midas_transform, midas_device = depth.init_depth(loader=init_depth_model)
        if midas_model is not None:
            print("Depth model ready")

    cv2.namedWindow("CarAlert", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("CarAlert", disp_w, disp_h)
    if ENABLE_DEPTH:
        cv2.namedWindow("Depth", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Depth", disp_w, disp_h)

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

        # depth (synchronous every N frames)
        if ENABLE_DEPTH and midas_model is not None and (frame_idx % DEPTH_EVERY_N_FRAMES == 0):
            raw = depth.predict_depth_sync(midas_model, midas_transform, midas_device, frame_t)
            if raw is not None:
                raw_vis = depth.visualize_depth(raw)
                cv2.imshow("Depth", cv2.resize(raw_vis, (disp_w, disp_h), interpolation=cv2.INTER_AREA))

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

"""
Detection module
Provides:
- detect_frame(model, frame, conf=None, iou=None, target_class='car') -> list of detections (x1,y1,x2,y2,conf)
- draw_detections(frame, detections, area_frac_threshold=0.3, target_class='car') -> (vis, alert_flag)

This module intentionally keeps a small API so it can be used by the new main app.
"""

import numpy as np
import cv2

# module-level defaults (will be overridden by caller when needed)
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45


def detect_frame(model, frame, conf=None, iou=None, target_class='car'):
    """Run detection on a single frame and return list of (x1,y1,x2,y2,conf).

    The function does not draw; it only returns detections for the target class.
    """
    if conf is None:
        conf = CONF_THRESHOLD
    if iou is None:
        iou = IOU_THRESHOLD

    detections = []
    try:
        results = model.predict(source=frame, conf=conf, iou=iou, verbose=False)
    except Exception:
        return detections

    if results and len(results) > 0:
        r = results[0]
        boxes = getattr(r, 'boxes', None)
        if boxes is not None and len(boxes) > 0:
            try:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                cls_ids = boxes.cls.cpu().numpy().astype(int)
            except Exception:
                xyxy = np.array(boxes.xyxy)
                confs = np.array(boxes.conf)
                cls_ids = np.array(boxes.cls).astype(int)
            # get names if available (model may expose names)
            try:
                names = [model.names.get(int(c), str(int(c))) for c in cls_ids]
            except Exception:
                names = [str(int(c)) for c in cls_ids]

            for (b, conf_val, name) in zip(xyxy, confs, names):
                if name.lower() == target_class.lower():
                    x1, y1, x2, y2 = map(int, b)
                    detections.append((x1, y1, x2, y2, float(conf_val)))
    return detections


def draw_detections(frame, detections, area_frac_threshold=0.3, target_class='car'):
    """Draw detections on a copy of `frame` and return (vis_frame, alert_flag).

    This function duplicates a small part of visual logic from the original script
    so the detection module is self-contained for drawing.
    """
    vis = frame.copy()
    frame_area = float(frame.shape[0] * frame.shape[1])
    alert_triggered = False
    for (x1, y1, x2, y2, conf_val) in detections:
        area = max(0, (x2 - x1) * (y2 - y1))
        frac = area / frame_area if frame_area > 0 else 0.0
        color = (0, 0, 255) if frac >= area_frac_threshold else (0, 200, 0)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"{target_class} {conf_val:.2f} {frac*100:.2f}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y_top = max(0, y1 - th - 6)
        cv2.rectangle(vis, (x1, y_top), (x1 + tw + 6, y_top + th + 6), color, -1)
        cv2.putText(vis, label, (x1 + 2, y_top + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        if frac >= area_frac_threshold:
            alert_triggered = True
    return vis, alert_triggered

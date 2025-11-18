"""
Depth module
Provides:
- init_depth() -> (midas_model, transform, device)
- predict_depth_sync(midas_model, transform, device, frame) -> raw_depth (2D numpy array)
- visualize_depth(raw) -> BGR uint8 image (normalized grayscale)

This module delegates model loading to the original `car_alert_simple` loader where appropriate.
"""

import numpy as np
import cv2

def init_depth(loader=None):
    """Load and return (midas_model, transform, device).

    If `loader` is provided, it should be a callable that returns the
    (midas_model, transform, device) tuple (this follows the original
    `init_depth_model` signature in `car_alert_simple`). If `loader` is None,
    this function returns (None, None, None).
    """
    if loader is None:
        return None, None, None
    try:
        return loader()
    except Exception:
        return None, None, None


def predict_depth_sync(midas_model, midas_transform, device, frame):
    """Run synchronous depth prediction on a BGR `frame` and return raw depth array.

    This mirrors the synchronous code path in the original script.
    """
    if midas_model is None or midas_transform is None:
        return None
    try:
        import torch
    except Exception:
        torch = None
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inp = midas_transform(img).to(device)
    if inp.dim() == 3:
        inp = inp.unsqueeze(0)
    with torch.no_grad():
        pred = midas_model(inp)
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        pred = torch.nn.functional.interpolate(pred, size=img.shape[:2], mode='bicubic', align_corners=False)
        depth = pred.squeeze().cpu().numpy()
    return depth


def visualize_depth(raw):
    """Normalize raw depth to 0..255 grayscale BGR image for display."""
    if raw is None:
        return None
    mask_finite = np.isfinite(raw)
    if not np.any(mask_finite):
        viz = np.zeros(raw.shape, dtype='uint8')
        return cv2.cvtColor(viz, cv2.COLOR_GRAY2BGR)
    dmin = float(np.nanmin(raw))
    dmax = float(np.nanmax(raw))
    if dmax - dmin > 1e-6:
        viz = (255.0 * (raw - dmin) / (dmax - dmin)).astype('uint8')
    else:
        viz = (raw * 0).astype('uint8')
    return cv2.cvtColor(viz, cv2.COLOR_GRAY2BGR)

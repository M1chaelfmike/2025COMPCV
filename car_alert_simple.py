"""
简化版车类警报脚本（独立文件）
功能：
- 只运行目标检测模型（Ultralytics YOLO）
- 在相机画面中只显示模型预测为 `car` 的检测框
- 当某个 car 的 bbox 面积占帧面积 >= area_frac（默认 0.30）时：
  - 将该 bbox 变为红色
  - 在控制台打印警告信息（包含占比与置信度）

用法示例：
python car_alert_simple.py --weights yolov10m.pt --camera 0 --conf 0.25 --area-frac 0.3

注意：请在已安装 ultralytics 与 opencv 的 Python 环境下运行。
"""

import time
import sys
import os

import cv2
import numpy as np
import threading
import queue

# --- ensure torch.hub cache uses project-local directory to avoid filling system drive ---
_PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
proj_cache = os.path.join(_PROJ_ROOT, 'torch_cache')
try:
    os.makedirs(proj_cache, exist_ok=True)
except Exception:
    pass
os.environ.setdefault('TORCH_HOME', proj_cache)
try:
    import torch as _torch_for_hub
    try:
        _torch_for_hub.hub.set_dir(proj_cache)
    except Exception:
        # older/newer torch may not have set_dir or it may fail; TORCH_HOME env var will be used
        pass
    try:
        print("torch.hub dir:", _torch_for_hub.hub.get_dir())
    except Exception:
        print("torch.hub directory set to", proj_cache)
except Exception:
    # torch may not be installed in this environment yet; env var still set for later
    print("torch not available at import time; TORCH_HOME set to", proj_cache)

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

# -------------------- 配置（在此修改参数，适合在 PyCharm 点击 Run） --------------------
WEIGHTS = "yolov10m.pt"   # 模型权重路径
CAMERA_INDEX = 3           # 摄像头索引（例如你之前用过 3）
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
AREA_FRAC_THRESHOLD = 0.3  # 占比阈值（0.3 即 30%）
DISPLAY_MAX_WIDTH = 1280

# 目标输入/显示分辨率（你要求的竖屏分辨率）
TARGET_WIDTH = 544
TARGET_HEIGHT = 960
# -----------------------------------------------------------------------------

# Depth model options
ENABLE_DEPTH = True            # whether to run MiDaS depth prediction
DEPTH_EVERY_N_FRAMES = 3      # enqueue depth prediction every N frames
USE_DEPTH_THREAD = True       # run depth model in a separate thread
DEPTH_MODEL_NAME = "MiDaS"  # options: MiDaS_small, MiDaS, DPT_Hybrid, DPT_Large
# Depth display / alerting modes
# If True, use an absolute depth threshold (raw depth values) to determine "near" pixels.
# If False, use percentile bins specified by BINS_PERCENTILES (existing behaviour).
DEPTH_USE_ABSOLUTE = False
# Absolute depth threshold (only used when DEPTH_USE_ABSOLUTE=True).
# NOTE: MiDaS outputs relative depth; this threshold makes sense only if you know the scale.
ABS_NEAR_THRESHOLD = 0.5

# depth worker globals (populated when depth is initialized)
_depth_queue = None
_depth_thread = None
_latest_depth_vis = None
_depth_lock = threading.Lock()
_last_detect_time = None
_last_depth_time = None
_last_depth_time_cached = None

# raw depth (float) kept for numeric checks (thread-safe via _depth_lock)
_latest_depth_raw = None

# depth-only 'too close' detection config (bins + threshold)
BINS_PERCENTILES = [33, 66]
BIN_COLORS = [ (0,0,255), (0,255,255), (255,0,0) ]  # near:red, mid:yellow, far:blue
TOO_CLOSE_PERCENT = 70  # percent (0-100)


def draw_label(frame, text, x, y, color=(0, 200, 0)):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    y_top = max(0, y - th - 6)
    cv2.rectangle(frame, (x, y_top), (x + tw + 6, y_top + th + 6), color, -1)
    cv2.putText(frame, text, (x + 2, y_top + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)


def init_model(weights_path: str):
    """Load YOLO model and return it. Prints warnings if weights missing."""
    if not os.path.exists(weights_path):
        print(f"Warning: weights file '{weights_path}' not found. Model load may fail if path is incorrect.")
    print(f"Loading model {weights_path} ...")
    model = YOLO(weights_path)
    print("Model loaded.")
    return model


def init_camera(index: int, target_w: int, target_h: int):
    """Open camera, try to set target resolution, and compute display size to fit DISPLAY_MAX_WIDTH.

    Returns: cap, disp_w, disp_h
    """
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"Cannot open camera {index}")
        sys.exit(2)

    # print original
    cam_w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    cam_h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    print(f"Camera native resolution before set: {cam_w0} x {cam_h0}")

    # try to set requested
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(target_w))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(target_h))

    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    print(f"Camera resolution after set attempt: {cam_w} x {cam_h}")

    # compute display size (fit to DISPLAY_MAX_WIDTH when wide)
    display_scale = 1.0
    if cam_w > DISPLAY_MAX_WIDTH and DISPLAY_MAX_WIDTH > 0:
        display_scale = DISPLAY_MAX_WIDTH / float(cam_w)
    disp_w = max(1, int(cam_w * display_scale))
    disp_h = max(1, int(cam_h * display_scale))
    return cap, disp_w, disp_h


def warmup_model(model):
    """Run one inference on a small image (test.jpg if available) to warm up the model.

    This helps avoid a long first-frame delay during capture loop.
    """
    test_path = 'test.jpg'
    if os.path.exists(test_path):
        img = cv2.imread(test_path)
        if img is None:
            # fallback
            img = np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8)
    else:
        img = np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8)

    try:
        print("Warming up model with a test image...")
        _ = model.predict(source=img, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)
        print("Warmup done.")
    except Exception as e:
        print("Warmup inference failed (ignored):", e)


def transform_frame_to_target(frame, target_w: int, target_h: int):
    """Scale then center-crop the input frame to exactly (target_w, target_h).

    This keeps aspect ratio and centers the crop.
    """
    h, w = frame.shape[:2]
    if (w, h) == (target_w, target_h):
        return frame

    # scale so that the scaled image covers the target area
    scale = max(target_w / w, target_h / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # center crop
    x0 = max(0, (new_w - target_w) // 2)
    y0 = max(0, (new_h - target_h) // 2)
    cropped = resized[y0:y0 + target_h, x0:x0 + target_w]
    # if crop is smaller due to rounding, pad
    ch, cw = cropped.shape[:2]
    if ch != target_h or cw != target_w:
        out = np.zeros((target_h, target_w, 3), dtype=frame.dtype)
        out[0:ch, 0:cw] = cropped
        return out
    return cropped


def update_and_draw(frame, model, target_class='car'):
    """Run detection on frame, draw only target_class boxes.

    Returns (vis_frame, alert_triggered: bool, detection_count:int)
    """
    frame_area = float(frame.shape[0] * frame.shape[1])
    # time the detection inference
    t0 = time.time()
    try:
        results = model.predict(source=frame, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)
    except Exception as e:
        print("Model inference failed:", e)
        return frame.copy(), False, 0, 0.0
    t1 = time.time()
    detect_time = (t1 - t0)
    # store last detect time (module-global)
    try:
        global _last_detect_time
        _last_detect_time = detect_time
    except Exception:
        pass

    car_detections = []  # list of (x1,y1,x2,y2,conf)
    if results and len(results) > 0:
        r = results[0]
        boxes = getattr(r, 'boxes', None)
        if boxes is not None and len(boxes) > 0:
            # boxes may be torch tensors; convert safely
            try:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                cls_ids = boxes.cls.cpu().numpy().astype(int)
            except Exception:
                # fallback if already numpy
                xyxy = np.array(boxes.xyxy)
                confs = np.array(boxes.conf)
                cls_ids = np.array(boxes.cls).astype(int)
            names = [model.names.get(int(c), str(int(c))) for c in cls_ids]

            for (b, conf_val, name) in zip(xyxy, confs, names):
                if name.lower() == target_class.lower():
                    x1, y1, x2, y2 = map(int, b)
                    car_detections.append((x1, y1, x2, y2, float(conf_val)))

    vis = frame.copy()
    alert_triggered = False
    for (x1, y1, x2, y2, conf_val) in car_detections:
        area = max(0, (x2 - x1) * (y2 - y1))
        frac = area / frame_area if frame_area > 0 else 0.0
        if frac >= AREA_FRAC_THRESHOLD:
            color = (0, 0, 255)  # red
            alert_triggered = True
        else:
            color = (0, 200, 0)  # green
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"{target_class} {conf_val:.2f} {frac*100:.2f}%"
        draw_label(vis, label, x1, y1, color=color)

    if alert_triggered and len(car_detections) > 0:
        # pick the largest for console message
        largest = max(car_detections, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        lx1, ly1, lx2, ly2, lconf = largest
        larea = max(0, (lx2 - lx1) * (ly2 - ly1))
        lfrac = larea / frame_area if frame_area > 0 else 0.0
        print(f"ALERT: {target_class} area fraction {lfrac*100:.1f}% (conf {lconf:.2f})")

    return vis, alert_triggered, len(car_detections), detect_time


def init_depth_model():
    """Load a MiDaS model by name using torch.hub and return (midas, transform, device).

    DEPTH_MODEL_NAME (top-level constant) controls which model to load. This avoids
    relying on an external MiDas.py and makes it easy to change models in this file.
    """
    try:
        import torch
    except Exception as e:
        print("Torch not available, cannot load depth model:", e)
        return None, None, None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device for depth:", device)

    model_name = DEPTH_MODEL_NAME
    try:
        print(f"Loading depth model '{model_name}' from torch.hub ...")
        # load transforms hub first
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        # pick transform based on model name
        lname = model_name.lower()
        if "small" in lname:
            transform = midas_transforms.small_transform
        elif "dpt" in lname or "hybrid" in lname:
            transform = midas_transforms.dpt_transform
        else:
            transform = midas_transforms.default_transform

        midas = torch.hub.load("intel-isl/MiDaS", model_name).to(device)
        midas.eval()

        # quick param count (approx)
        try:
            params = sum(p.numel() for p in midas.parameters())
            print(f"Loaded {model_name} with ~{params/1e6:.2f}M parameters")
        except Exception:
            pass

        # warm one inference on a zero image to measure single-frame time
        try:
            w, h = TARGET_WIDTH, TARGET_HEIGHT
            img = np.zeros((h, w, 3), dtype=np.uint8)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            inp = transform(img_rgb).to(device)
            if inp.dim() == 3:
                inp = inp.unsqueeze(0)
            import time as _tt
            with torch.no_grad():
                t0 = _tt.time()
                _ = midas(inp)
                t1 = _tt.time()
            print(f"Depth warm-run time for {model_name}: {(t1-t0)*1000:.0f} ms")
        except Exception as e:
            print("Depth warm-run failed (ignored):", e)

        return midas, transform, device
    except Exception as e:
        print("Failed to load depth model from torch.hub:", e)
        # fallback: try importing project MiDas.py
        try:
            import MiDas as _m
            midas = getattr(_m, 'midas', None)
            transform = getattr(_m, 'transform', None)
            device = getattr(_m, 'device', None)
            if midas is None or transform is None:
                raise ImportError("MiDas module does not expose required symbols")
            print("Fallback: Depth model imported from MiDas.py")
            return midas, transform, device
        except Exception as e2:
            print("Fallback import of MiDas.py failed:", e2)
            return None, None, None


def compute_depth_bins(depth: np.ndarray, percentiles: 'list[int]'):
    """Split depth into bins using percentile thresholds.

    Returns (bins, thresholds)
    """
    d = np.array(depth, dtype=np.float32)
    mask_finite = np.isfinite(d)
    if not np.any(mask_finite):
        return [], []
    flat = d[mask_finite]
    thresholds = [float(np.percentile(flat, p)) for p in percentiles]
    bins = []
    if len(thresholds) == 0:
        bins.append(mask_finite.copy())
        return bins, thresholds
    t0 = thresholds[0]
    bins.append(mask_finite & (d <= t0))
    for i in range(1, len(thresholds)):
        lo = thresholds[i-1]
        hi = thresholds[i]
        bins.append(mask_finite & (d > lo) & (d <= hi))
    bins.append(mask_finite & (d > thresholds[-1]))
    return bins, thresholds


def make_bin_visual(bins: 'list[np.ndarray]', colors: 'list[tuple]', shape):
    vis = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    for m, col in zip(bins, colors):
        mask = (m.astype(bool))
        vis[mask] = col
    return vis


def compute_region_area(raw: np.ndarray, *, mask: 'np.ndarray | None' = None, abs_thresh: float | None = None, percentile: float | None = None, compare_le: bool = True):
    """Compute pixel count and percentage of a region in `raw`.

    Parameters:
    - raw: 2D depth array (float)
    - mask: optional boolean mask to limit region (same shape as raw); only finite pixels inside mask considered
    - abs_thresh: if provided, uses an absolute threshold and returns pixels where raw <= abs_thresh (if compare_le=True) or >= abs_thresh (if False)
    - percentile: if provided (and abs_thresh is None), compute threshold = percentile of finite values and apply same compare
    - compare_le: when True use <= threshold as "in region", else use >=

    Returns: (region_count, total_count, percent, threshold_used)
    """
    if raw is None:
        return 0, 0, 0.0, None
    mask_finite = np.isfinite(raw)
    if mask is not None:
        try:
            combined = mask_finite & (mask.astype(bool))
        except Exception:
            combined = mask_finite
    else:
        combined = mask_finite

    total = int(np.count_nonzero(combined))
    if total == 0:
        return 0, 0, 0.0, None

    thr = None
    if abs_thresh is not None:
        thr = float(abs_thresh)
    elif percentile is not None:
        flat = raw[mask_finite]
        if flat.size > 0:
            thr = float(np.percentile(flat, float(percentile)))

    if thr is None:
        # nothing to threshold against
        return 0, total, 0.0, thr

    if compare_le:
        region = combined & (raw <= thr)
    else:
        region = combined & (raw >= thr)

    region_count = int(np.count_nonzero(region))
    pct = (region_count / total) * 100.0 if total > 0 else 0.0
    return region_count, total, pct, thr


def _depth_worker_loop(midas, transform, device, q: 'queue.Queue'):
    """Worker loop that consumes frames from q and updates _latest_depth_vis.

    Each item is expected to be a BGR numpy array (frame). Put None to stop.
    """
    global _latest_depth_vis, _latest_depth_raw, _last_depth_time
    try:
        import torch
    except Exception:
        torch = None

    global _last_depth_time
    while True:
        try:
            item = q.get()
        except Exception:
            break
        if item is None:
            break
        frame = item
        try:
            dt0 = time.time()
            # preprocess: BGR -> RGB
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            inp = transform(img).to(device)
            # ensure batch dim
            if inp.dim() == 3:
                inp = inp.unsqueeze(0)
            with torch.no_grad():
                pred = midas(inp)
                # pred: [1,1,Hf,Wf] or [1,Hf,Wf]
                if pred.dim() == 3:
                    pred = pred.unsqueeze(1)
                # interpolate to original image size
                pred = torch.nn.functional.interpolate(pred, size=img.shape[:2], mode='bicubic', align_corners=False)
                depth = pred.squeeze().cpu().numpy()

            # normalize to uint8 for visualization
            dmin, dmax = depth.min(), depth.max()
            if dmax - dmin > 1e-6:
                viz = (255 * (depth - dmin) / (dmax - dmin)).astype('uint8')
            else:
                viz = (depth * 0).astype('uint8')

            # colorize single-channel depth for nicer view
            depth_color = cv2.applyColorMap(viz, cv2.COLORMAP_MAGMA)

            # update shared visuals, raw depth and time atomically under the lock
            try:
                with _depth_lock:
                    _latest_depth_vis = depth_color
                    # store a copy of raw depth for numeric checks/alerts
                    try:
                        _latest_depth_raw = depth.copy()
                    except Exception:
                        _latest_depth_raw = np.array(depth)
                    _last_depth_time = time.time() - dt0
                print(f"Depth produced: {(_last_depth_time*1000):.0f} ms")
            except Exception:
                pass
        except Exception as e:
            print("Depth worker error (ignored):", e)
            continue


def stop_depth_worker():
    global _depth_queue, _depth_thread, _latest_depth_vis, _last_depth_time_cached, _last_depth_time
    if _depth_queue is not None:
        try:
            _depth_queue.put(None, block=False)
        except Exception:
            try:
                _depth_queue.put(None)
            except Exception:
                pass
    if _depth_thread is not None:
        _depth_thread.join(timeout=2.0)
        _depth_thread = None


def main():
    # 使用顶部常量配置，按模块拆分函数实现初始化和主循环
    if YOLO is None:
        print("Error: ultralytics YOLO is not available in this environment. Please install it first.")
        sys.exit(2)

    model = init_model(WEIGHTS)
    cap, disp_w, disp_h = init_camera(CAMERA_INDEX, TARGET_WIDTH, TARGET_HEIGHT)

    # 模型热启动：使用 test.jpg（若存在）或空图像做一次推理，确保模型内部准备完毕
    warmup_model(model)

    # initialize depth model and worker (optional)
    midas_model = None
    midas_transform = None
    midas_device = None
    global _depth_queue, _depth_thread, _latest_depth_vis, _latest_depth_raw, _last_depth_time, _last_depth_time_cached
    if ENABLE_DEPTH:
        midas_model, midas_transform, midas_device = init_depth_model()
        if midas_model is not None:
            _depth_queue = queue.Queue(maxsize=4)
            if USE_DEPTH_THREAD:
                _depth_thread = threading.Thread(target=_depth_worker_loop, args=(midas_model, midas_transform, midas_device, _depth_queue), daemon=True)
                _depth_thread.start()
                print("Depth worker started (thread)")
                # try to enqueue one initial frame to kick off the worker so it produces first depth quickly
                try:
                    ret0, f0 = cap.read()
                    if ret0:
                        f0t = transform_frame_to_target(f0, TARGET_WIDTH, TARGET_HEIGHT)
                        try:
                            _depth_queue.put_nowait(f0t.copy())
                            print("Enqueued initial frame for depth warmup")
                        except Exception:
                            try:
                                _depth_queue.put(f0t.copy(), timeout=0.5)
                                print("Enqueued initial frame for depth warmup (blocking)")
                            except Exception:
                                print("Failed to enqueue initial depth warmup frame")
                except Exception:
                    print("Could not read initial frame for depth warmup")

    cv2.namedWindow("CarAlert", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("CarAlert", disp_w, disp_h)
    if ENABLE_DEPTH:
        cv2.namedWindow("Depth", cv2.WINDOW_NORMAL)
        # depth display size: same as car display for simplicity
        cv2.resizeWindow("Depth", disp_w, disp_h)

    print("Initialization complete. Press 'q' to quit.")

    frame_idx = 0
    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera")
            break

        # 强制将捕获帧变换为目标尺寸（保持内容居中、裁切或填充）
        frame_t = transform_frame_to_target(frame, TARGET_WIDTH, TARGET_HEIGHT)

        vis, alert_triggered, det_count, detect_time = update_and_draw(frame_t, model, target_class='car')
        # print inference times for both models (if available)
        # read depth time under lock to avoid race with worker
        try:
            dt_ms = detect_time * 1000.0
        except Exception:
            dt_ms = None
        with _depth_lock:
            dpt = _last_depth_time
        # keep a cached last-known depth time so main can display a value even
        # if worker just completed after this print. Update cache when new value arrives.
        dpt_cached = globals().get('_last_depth_time_cached', None)
        if dpt is not None:
            globals()['_last_depth_time_cached'] = dpt
        else:
            dpt = dpt_cached
        try:
            dpt_ms = dpt * 1000.0 if dpt is not None else None
        except Exception:
            dpt_ms = None
        print(f"Detect: {dt_ms:.0f} ms" if dt_ms is not None else "Detect: -", end='')
        print(f" | Depth: {dpt_ms:.0f} ms" if dpt_ms is not None else " | Depth: -")

        # enqueue or run depth prediction
        if ENABLE_DEPTH and midas_model is not None:
            if frame_idx % DEPTH_EVERY_N_FRAMES == 0:
                if USE_DEPTH_THREAD and _depth_queue is not None:
                    try:
                        _depth_queue.put_nowait(frame_t.copy())
                    except Exception:
                        # queue full, skip this frame
                        pass
                else:
                    # synchronous depth inference
                    try:
                        dt0 = time.time()
                        import torch
                        img = cv2.cvtColor(frame_t, cv2.COLOR_BGR2RGB)
                        inp = midas_transform(img).to(midas_device)
                        if inp.dim() == 3:
                            inp = inp.unsqueeze(0)
                        with torch.no_grad():
                            pred = midas_model(inp)
                            if pred.dim() == 3:
                                pred = pred.unsqueeze(1)
                            pred = torch.nn.functional.interpolate(pred, size=img.shape[:2], mode='bicubic', align_corners=False)
                            depth = pred.squeeze().cpu().numpy()
                        dmin, dmax = depth.min(), depth.max()
                        if dmax - dmin > 1e-6:
                            viz = (255 * (depth - dmin) / (dmax - dmin)).astype('uint8')
                        else:
                            viz = (depth * 0).astype('uint8')
                        depth_color = cv2.applyColorMap(viz, cv2.COLORMAP_MAGMA)
                        try:
                            with _depth_lock:
                                _latest_depth_vis = depth_color
                                try:
                                    _latest_depth_raw = depth.copy()
                                except Exception:
                                    _latest_depth_raw = np.array(depth)
                                _last_depth_time = time.time() - dt0
                        except Exception:
                            pass
                    except Exception as e:
                        print("Synchronous depth error (ignored):", e)

        # 如需打印性能信息
        # t1 = time.time(); print(f"Frame time: {(t1-t0)*1000:.0f} ms, detections: {det_count}")

        # scale for display
        vis_disp = cv2.resize(vis, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
        cv2.imshow("CarAlert", vis_disp)

        # show latest depth bins (if available) and depth-only 'too close' alert
        if ENABLE_DEPTH:
            with _depth_lock:
                raw = None if _latest_depth_raw is None else _latest_depth_raw.copy()
            if raw is not None:
                try:
                    # Show the original raw depth prediction (normalized for display)
                    # and print numeric output statistics to console.
                    mask_finite = np.isfinite(raw)
                    finite_total = int(np.count_nonzero(mask_finite))

                    if finite_total > 0:
                        dmin = float(np.nanmin(raw))
                        dmax = float(np.nanmax(raw))
                        dmean = float(np.nanmean(raw))
                    else:
                        dmin = dmax = dmean = 0.0

                    # sample center value (for quick debugging)
                    try:
                        h, w = raw.shape
                        center_val = float(raw[h // 2, w // 2]) if np.isfinite(raw[h // 2, w // 2]) else float('nan')
                    except Exception:
                        center_val = float('nan')

                    # Print raw output stats each frame (concise)
                    print(f"Depth raw stats: min={dmin:.3f}, max={dmax:.3f}, mean={dmean:.3f}, finite_pixels={finite_total}, center={center_val:.3f}")

                    # Compute region area (either absolute-threshold or percentile) and overlay
                    if DEPTH_USE_ABSOLUTE:
                        region_count, region_total, region_pct, thr = compute_region_area(raw, abs_thresh=ABS_NEAR_THRESHOLD)
                        mode_label = f"ABS"
                    else:
                        # use first percentile value from BINS_PERCENTILES as region threshold
                        pct = BINS_PERCENTILES[0] if len(BINS_PERCENTILES) > 0 else 33
                        region_count, region_total, region_pct, thr = compute_region_area(raw, percentile=pct)
                        mode_label = f"P{pct}"

                    # Normalize raw depth to 0..255 for display (grayscale)
                    if dmax - dmin > 1e-6:
                        viz = (255.0 * (raw - dmin) / (dmax - dmin)).astype('uint8')
                    else:
                        viz = (raw * 0).astype('uint8')

                    raw_vis = cv2.cvtColor(viz, cv2.COLOR_GRAY2BGR)
                    # Overlay stats and region percentage on the visualization
                    txt = f"min={dmin:.1f} max={dmax:.1f} mean={dmean:.2f}"
                    cv2.putText(raw_vis, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                    if thr is not None:
                        txt2 = f"Region({mode_label}): {region_pct:.1f}% thr={thr:.1f}"
                    else:
                        txt2 = f"Region({mode_label}): {region_pct:.1f}%"
                    cv2.putText(raw_vis, txt2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

                    # Print concise region info to console
                    print(f"Region area ({mode_label}) -> {region_count}/{region_total} = {region_pct:.2f}% thr={thr}")

                    dv_disp = cv2.resize(raw_vis, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
                    cv2.imshow("Depth", dv_disp)
                except Exception as e:
                    print("Depth display error:", e)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q')):
            break

    cap.release()
    # stop depth worker cleanly
    if ENABLE_DEPTH and _depth_thread is not None:
        stop_depth_worker()
        # close depth window
        try:
            cv2.destroyWindow('Depth')
        except Exception:
            pass
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

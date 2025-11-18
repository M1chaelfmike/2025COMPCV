"""
Depth prediction helper script using MiDaS (intel-isl/MiDaS).

Usage:
- Edit the constants at top or pass an image path as the first CLI argument.
- The script will load the chosen MiDaS model, run inference on the image,
  save a colorized depth visualization and a raw .npy depth map, and show the result.

Notes:
- Downloads are cached under the project-local torch cache if available.
- Designed to be run from PyCharm (no required CLI args) or from command line.
"""

import os
import sys
import time

import cv2
import numpy as np

# Configuration (edit here if you run via PyCharm)
INPUT_IMAGE = 'test.jpg'                   # default image path (relative to script)
MODEL_NAME = 'MiDaS'                 # options: MiDaS_small, MiDaS, DPT_Hybrid, DPT_Large
OUTPUT_VIS_PNG = 'depth_vis.png'           # colorized depth output
OUTPUT_DEPTH_NPY = 'depth_raw.npy'         # raw float32 depth map
PROJECT_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'torch_cache')  # where to cache weights
DEVICE = 'cuda' if (os.getenv('CUDA_VISIBLE_DEVICES', None) is not None or False) else None

# Binning / proportions: percentiles used to split depth into N+1 bins.
# Example: BINS_PERCENTILES = [33, 66] -> three bins: <=33pct, 33-66, >=66
BINS_PERCENTILES = [33, 66]
# Colors for discrete bin visualization (BGR tuples). Length should be len(BINS_PERCENTILES)+1
BIN_COLORS = [ (0,0,255), (0,255,255), (255,0,0) ]  # near:red, mid:yellow, far:blue
# Too-close alert threshold: if the 'near' bin occupies >= TOO_CLOSE_PERCENT of image -> alert
TOO_CLOSE_PERCENT = 70  # percent (0-100)

# Try to set DEVICE more robustly after importing torch


def ensure_torch_cache(path):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass
    os.environ.setdefault('TORCH_HOME', path)
    try:
        import torch as _t
        try:
            _t.hub.set_dir(path)
        except Exception:
            pass
    except Exception:
        # torch might not be installed; user will see import error later
        pass


def load_midas(model_name: str):
    """Load MiDaS model and appropriate transform using torch.hub.

    Returns (midas_model, transform, device)
    """
    try:
        import torch
    except Exception as e:
        raise RuntimeError('torch is required for MiDaS but not installed') from e

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using device for depth model:', device)

    print(f"Loading depth model '{model_name}' from intel-isl/MiDaS (torch.hub)...")
    # load transforms first
    midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
    lname = model_name.lower()
    if 'small' in lname:
        transform = midas_transforms.small_transform
    elif 'dpt' in lname or 'hybrid' in lname:
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.default_transform

    midas = torch.hub.load('intel-isl/MiDaS', model_name).to(device)
    midas.eval()

    # print parameter count if possible
    try:
        params = sum(p.numel() for p in midas.parameters())
        print(f"Loaded {model_name}, params: {params:,} (~{params/1e6:.2f}M)")
    except Exception:
        pass

    return midas, transform, device


def predict_depth(midas, transform, device, img_bgr: np.ndarray):
    """Run MiDaS on a BGR image and return depth (float32 HxW) and timing (s).

    The returned depth is the raw model output (relative depth). It is not in metres.
    """
    import torch

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    inp = transform(img_rgb).to(device)
    if inp.dim() == 3:
        inp = inp.unsqueeze(0)

    with torch.no_grad():
        t0 = time.time()
        pred = midas(inp)
        t1 = time.time()

        # normalize dims and interpolate to original image size
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        pred = torch.nn.functional.interpolate(pred, size=img_rgb.shape[:2], mode='bicubic', align_corners=False)
        depth = pred.squeeze().cpu().numpy()

    return depth.astype(np.float32), (t1 - t0)


def normalize_depth_for_vis(depth: np.ndarray):
    """Convert depth float map to uint8 color map for visualization."""
    dmin, dmax = float(np.nanmin(depth)), float(np.nanmax(depth))
    if dmax - dmin < 1e-6:
        vis = np.zeros(depth.shape, dtype=np.uint8)
    else:
        vis = (255.0 * (depth - dmin) / (dmax - dmin)).clip(0, 255).astype(np.uint8)
    vis_color = cv2.applyColorMap(vis, cv2.COLORMAP_MAGMA)
    return vis_color, dmin, dmax


def compute_depth_bins(depth: np.ndarray, percentiles: 'list[int]'):
    """Split depth into bins using percentile thresholds.

    Returns: (bin_masks, thresholds)
      - bin_masks: list of boolean masks for each bin (same shape as depth)
      - thresholds: list of threshold values (len == len(percentiles))
    """
    d = np.array(depth, dtype=np.float32)
    mask_finite = np.isfinite(d)
    if not np.any(mask_finite):
        return [], []

    flat = d[mask_finite]
    thresholds = [float(np.percentile(flat, p)) for p in percentiles]
    bins = []
    # first bin: <= t0
    if len(thresholds) == 0:
        bins.append(mask_finite.copy())
        return bins, thresholds

    t0 = thresholds[0]
    bins.append(mask_finite & (d <= t0))
    for i in range(1, len(thresholds)):
        lo = thresholds[i-1]
        hi = thresholds[i]
        bins.append(mask_finite & (d > lo) & (d <= hi))
    # last bin: > last threshold
    bins.append(mask_finite & (d > thresholds[-1]))
    return bins, thresholds


def make_bin_visual(bins: 'list[np.ndarray]', colors: 'list[tuple]', shape):
    """Create a BGR image where each bin mask is colored by corresponding color."""
    vis = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    for m, col in zip(bins, colors):
        # ensure mask boolean
        mask = (m.astype(bool))
        vis[mask] = col
    return vis


def main():
    # set cache
    ensure_torch_cache(PROJECT_CACHE)

    # allow CLI override of image path and model
    img_path = INPUT_IMAGE
    model_name = MODEL_NAME
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    if len(sys.argv) > 2:
        model_name = sys.argv[2]

    if not os.path.exists(img_path):
        print(f"Input image not found: {img_path}")
        print("Place an image at the path or pass the image path as first argument.")
        return

    img = cv2.imread(img_path)
    if img is None:
        print("Failed to read image (cv2.imread returned None)")
        return

    # load model
    midas, transform, device = load_midas(model_name)

    print("Running depth prediction...")
    depth, t_infer = predict_depth(midas, transform, device, img)

    vis_color, dmin, dmax = normalize_depth_for_vis(depth)
    print(f"Depth raw stats: min={dmin:.4f}, max={dmax:.4f}, infer_time={(t_infer*1000):.0f} ms")

    # save outputs (colorized and raw)
    try:
        cv2.imwrite(OUTPUT_VIS_PNG, vis_color)
        np.save(OUTPUT_DEPTH_NPY, depth)
        print(f"Saved visualization to {OUTPUT_VIS_PNG} and raw depth to {OUTPUT_DEPTH_NPY}")
    except Exception as e:
        print("Failed to save outputs:", e)

    # compute percentile bins and print proportions
    bins, thresholds = compute_depth_bins(depth, BINS_PERCENTILES)
    if len(bins) == 0:
        print("No finite depth values to analyze.")
    else:
        counts = [int(np.count_nonzero(b)) for b in bins]
        total = sum(counts)
        pct = [ (c / total * 100.0) if total > 0 else 0.0 for c in counts ]
        print("Depth bins (by percentiles):")
        for i, c in enumerate(counts):
            if i == 0:
                rng = f"<= {BINS_PERCENTILES[0]}pct (<= {thresholds[0]:.4f})"
            elif i <= len(thresholds)-1:
                rng = f"{BINS_PERCENTILES[i-1]}-{BINS_PERCENTILES[i]}pct ({thresholds[i-1]:.4f}-{thresholds[i]:.4f})"
            else:
                rng = f"> {BINS_PERCENTILES[-1]}pct (> {thresholds[-1]:.4f})"
            print(f"  bin {i}: {rng} -> {counts[i]} px, {pct[i]:.2f}%")

    # show only the discrete-bin visualization and issue an alert if near-bin占比 >= TOO_CLOSE_PERCENT
    try:
        if len(bins) > 0:
            cols = BIN_COLORS[:len(bins)] if len(BIN_COLORS) >= len(bins) else (BIN_COLORS * ((len(bins) // len(BIN_COLORS)) + 1))[:len(bins)]
            bin_vis = make_bin_visual(bins, cols, vis_color.shape[:2])

            # compute near bin percent
            near_count = int(np.count_nonzero(bins[0]))
            total = vis_color.shape[0] * vis_color.shape[1]
            near_pct = (near_count / total) * 100.0 if total > 0 else 0.0

            # if too close, overlay a red warning text onto bin_vis
            if near_pct >= float(TOO_CLOSE_PERCENT):
                txt = f"TOO CLOSE: near {near_pct:.1f}% >= {TOO_CLOSE_PERCENT}%"
                cv2.putText(bin_vis, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
                print("ALERT: TOO CLOSE ->", txt)
            else:
                txt = f"Near: {near_pct:.1f}%"
                cv2.putText(bin_vis, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow('Depth (bins)', bin_vis)
        else:
            print('No depth bins available to visualize')
        print("Press any key on the image window to exit")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception:
        print('Display not available in this environment. Exiting.')


if __name__ == '__main__':
    main()

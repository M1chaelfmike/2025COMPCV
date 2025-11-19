

中文说明：

本仓库包含一个使用 YOLO 和 OpenCV 的简易车辆检测演示。同时包含一个小型 Flask 服务（`flask_server.py`），用于将 `main_app` 里的可视化帧以 MJPEG 流形式对外提供，并提供一个简单的网页用于查看。

主要文件：
- `main_app.py` — 推理主循环（使用 YOLO），会打开 OpenCV 窗口并运行检测。
- `flask_server.py` — 在后台线程启动 Flask 并在主线程运行 `main_app.main()`；对外提供 `/` 和 `/stream`。
- `detection.py` — `main_app` 使用的检测工具函数。
- `yolov10m.pt` — 模型权重（应放在仓库根目录）。

分辨率与显示说明：
- `main_app.py` 定义了 `TARGET_WIDTH` 与 `TARGET_HEIGHT`（默认：544×960）。在进行检测前，帧会被变换为该目标分辨率。
- `flask_server.py` 提供的网页会按 `TARGET_WIDTH` 等比缩放显示 MJPEG 流，页面显示与处理分辨率一致或为等比缩放版本。

使用 OBS Studio（测试与虚拟摄像头）：
- 为了复现应用场景，本仓库演示中使用了 OBS Studio。我们将 `test.mp4` 导入为 OBS 的媒体源，并通过 OBS 的虚拟摄像头（Virtual Camera）输出，作为脚本的摄像头输入来源。
- 因此脚本中的分辨率参数（例如 `TARGET_WIDTH` / `TARGET_HEIGHT`）已根据该视频进行了修改；后续如果更换视频源，请相应地调整这些分辨率参数以匹配视频的实际分辨率或保持等比缩放。

---


环境与依赖（Windows PowerShell）：

推荐 Python 版本：3.8 及以上（建议 3.10/3.11）。

1) 创建并激活虚拟环境（PowerShell）：

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
```

2) 安装依赖（使用仓库中的 `requirements.txt`）：

```powershell
python -m pip install -U pip
python -m pip install -r requirements.txt
```

关于大型包的说明：
- `torch` / GPU：若有 CUDA GPU 并希望使用 GPU 加速，请按照 https://pytorch.org/get-started/locally/ 的官方说明安装对应的 `torch`。`requirements.txt` 中刻意未包含 `torch`，以便你为你的平台选择合适的 wheel。
- `ultralytics`：提供 YOLO 的接口；如不安装模型也可启动服务器，但检测会失败。

3) 确认 `yolov10m.pt` 位于项目根目录。若使用其他权重，修改 `main_app.py` 中的 `WEIGHTS` 或将权重重命名为 `yolov10m.pt`。

运行示例（单命令启动）：

在项目根目录执行（PowerShell）：

```powershell
python .\\flask_server.py
```

此命令行为：
- 在后台线程启动 Flask（监听端口 5000）。
- 在主线程运行 `main_app.main()`：打开名为 `CarAlert` 的 OpenCV 窗口，开始读取摄像头帧、执行检测，并发布可视化帧。

在浏览器打开 http://127.0.0.1:5000/ 查看网页及实时 MJPEG 流，流地址为 http://127.0.0.1:5000/stream（MJPEG）。

停止方式：
- 在 OpenCV 窗口按 `q` 关闭，或在控制台按 Ctrl+C 停止整个进程。

单独运行组件：
- 仅运行推理（不启动 Web 服务）：

```powershell
python .\\main_app.py
```

- 仅运行 Flask（前提是你单独让 `main_app` 发布帧或改为独立服务）：

```powershell
python -m flask run --host=0.0.0.0 --port=5000
```

配置与故障排查：
- 摄像头索引：若你的摄像头索引不是 `3`，请修改 `main_app.py` 顶部的 `CAMERA_INDEX`。
- 导入失败：若 `ultralytics` 或 `torch` 导入失败，请在虚拟环境中安装对应包；PyTorch 请按官方选择合适的安装命令。
- 模型权重：若找不到权重文件，`main_app` 会打印警告，请确认权重文件路径正确。
- 防火墙：Windows 可能会提示允许 Python 接受网络连接，按需允许或仅绑定 localhost。

开发者说明：
- `main_app` 提供线程安全的 `get_latest_frame()`，供 `flask_server.py` 读取最新可视化帧并编码为 JPEG。
- Flask 在后台线程启动，`main_app.main()` 在主线程运行以保证 OpenCV GUI 在 Windows 上正常工作。

如果你希望我继续改进：
- 我可以把 `requirements.txt` 写得更完整并添加 `setup` 脚本以自动化虚拟环境与依赖安装。
- 我可以改进前端（自动适配浏览器大小并显示帧率）。
- 我可以添加将 MJPEG 流保存为文件的选项。

---

English (below):

# Car Alert / Detection — README

This repository contains a simple vehicle-detection demo using YOLO and OpenCV.
It also includes a small Flask server (`flask_server.py`) that exposes the latest
visualisation frame from `main_app` as an MJPEG stream and a minimal web page to view it.

**Key files**
- `main_app.py` — main inference loop using YOLO (opens an OpenCV window and runs detection).
- `flask_server.py` — starts a Flask server (background thread) and runs `main_app.main()` in the main thread; serves `/` and `/stream`.
- `detection.py` — detection utilities used by `main_app`.
- `yolov10m.pt` — model weights (expected in repo root).

**Notes about resolution and display**
- `main_app.py` defines `TARGET_WIDTH` and `TARGET_HEIGHT` (default: 544x960). The inference frames are transformed to this target resolution before running detection.
- The web page served by `flask_server.py` displays the MJPEG stream scaled to `TARGET_WIDTH` while preserving aspect ratio, so the web view matches (or is an equal-scale version of) the processing resolution.

OBS Studio (test and virtual camera):
- To reproduce the demo scenario we use OBS Studio. Import `test.mp4` into OBS as a media source and enable OBS Virtual Camera to provide a virtual camera device for the scripts to consume.
- The script resolution parameters (e.g. `TARGET_WIDTH` / `TARGET_HEIGHT`) were adjusted to match that video; if you change the video source later, update those resolution parameters accordingly or keep an appropriate scale.

---

**Environment & Dependencies (Windows PowerShell)**

Recommended Python: 3.8+ (3.10/3.11 tested preferred).

1) Create and activate a virtual environment (PowerShell):

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
```

2) Install dependencies from the provided `requirements.txt`:

```powershell
python -m pip install -U pip
python -m pip install -r requirements.txt
```

Notes on heavy packages:
- `torch` / GPU: If you have a CUDA-capable GPU and want GPU acceleration, install `torch` following the official instructions at https://pytorch.org/get-started/locally/ (choose the right CUDA version). The `requirements.txt` intentionally omits `torch` so you can install the appropriate wheel manually.
- `ultralytics` is required for the YOLO interface; if you only want to run a stripped-down test without the model, the server will still start but detection will fail.

3) Verify `yolov10m.pt` exists in the project root. If you use another weights file, update `WEIGHTS` in `main_app.py` or place your weights as `yolov10m.pt`.

---

**Run the demo (single command)**

From project root (PowerShell):

```powershell
python .\\flask_server.py
```

What this does:
- Starts the Flask server in a background thread (listening on port 5000).
- Runs `main_app.main()` in the main thread — this opens an OpenCV window named `CarAlert` and begins reading frames from the camera, running detection, and publishing visualization frames.

Open the browser to http://127.0.0.1:5000/ to view the web page and live MJPEG stream. The stream endpoint is at http://127.0.0.1:5000/stream (MJPEG).

To stop:
- Close the OpenCV window and press `q` (or close the console and Ctrl+C to stop Flask/main process).

---

**Running components separately**
- Run the inference app only (no web server):

```powershell
python .\\main_app.py
```

- Run only Flask server (if you change code to run main_app separately and publish frames using the callback):
```powershell
python -m flask run --host=0.0.0.0 --port=5000
```

---

**Configuration & Troubleshooting**
- Camera index: If your camera is not at index `3`, change `CAMERA_INDEX` in `main_app.py` (top of the file).
- Missing packages / import errors: If `ultralytics` or `torch` imports fail, install them in the virtual environment. For PyTorch, follow the official installer command appropriate for your platform and CUDA version.
- Model weights: If the weights file cannot be found, `main_app` prints a warning. Ensure `yolov10m.pt` is in the project root or update `WEIGHTS` in `main_app.py`.
- Firewall: Windows may prompt to allow Python to accept incoming connections when Flask starts. Allow it (or restrict to localhost if preferred).

**Developer notes**
- `main_app` exposes a thread-safe `get_latest_frame()` helper used by `flask_server.py` to read the latest visualization frame and serve it as JPEG.
- The Flask server is intentionally started in a daemon thread and `main_app.main()` runs in the main thread so OpenCV GUI works correctly on Windows.

---

If you'd like, I can:
- Add a richer `requirements.txt` and a `setup` script to automate venv + install steps.
- Add a small JS frontend that adaptively fits the video to the browser window and displays FPS.
- Add an option to save the MJPEG stream to disk.

Feel free to tell me which of those you'd like next.



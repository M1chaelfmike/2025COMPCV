# CVproject-CV

说明（中文）

本仓库包含一个简化的车辆警报脚本 `car_alert_simple.py`，功能摘要：

<!-- Demo image: place a file named demo.png in the repo root to show here. -->
![Demo image (demo.png)](demo.png)


- 使用 Ultralytics YOLO 进行目标检测（示例中只对 `car` 显示框）。
- 可选地使用 MiDaS 深度模型进行深度预测，并在 `Depth` 窗口显示深度灰度图。
- 新增了 `compute_region_area` 帮助函数，可按绝对阈值或百分位计算某区域（或掩码内）像素数与占比，并在界面与控制台输出。

注意：深度模块（MiDaS）功能在本仓库中已实现（depth prediction + grayscale visualization + region area calculation）。

预警功能（alerting）说明：

- 当前脚本中针对物体是否 "太近" 的报警逻辑仍为原型/未完善状态：
  - 脚本原先使用百分位分箱（`BINS_PERCENTILES = [33,66]`），这会根据当前帧像素分布把大约 33% 的像素标为“near”，因此百分比看起来常常接近 33%。
  - 已加入可选的绝对阈值模式（`DEPTH_USE_ABSOLUTE = True` 与 `ABS_NEAR_THRESHOLD`），但 MiDaS 输出为相对深度/视差，直接用绝对阈值需要先做尺度标定或现场试验确定合适阈值。
  - 建议：如需可靠报警，应使用相机标定与尺度恢复或采样已知距离目标来选择阈值，或在检测到 `car` 时仅统计 bbox 内的近像素占比作为触发条件。

如何运行（基本）：

1. 创建并激活包含依赖的 Python 环境，安装依赖：

```powershell
pip install -r requirements.txt
# 或至少安装 ultralytics, opencv-python, torch 等
pip install ultralytics opencv-python
```

2. 运行示例脚本：

```powershell
python car_alert_simple.py --weights yolov10m.pt --camera 0
```

推送到 GitHub（常见步骤）

- 如果本地尚未初始化 git 仓库：

```powershell
cd "e:/Documents/PolyU/CV/CVproject-CV"
# 初始化仓库（如果已是仓库可跳过）
git init
# 检查当前状态
git status
# 添加文件并提交
git add README.md car_alert_simple.py
git commit -m "Add README; depth visualization and region-area helper"
```

- 在 GitHub 上创建一个新仓库（可以使用网页或 `gh` CLI）。假设远程仓库 URL 为 `https://github.com/USERNAME/REPO.git`，将本地仓库与远程关联并推送：

```powershell
# 如果需要重命名主分支为 main（可选）
git branch -M main
git remote add origin https://github.com/USERNAME/REPO.git
# 首次推送并设置上游分支
git push -u origin main
```

- 如果你使用 SSH（并且已配置 SSH key）：

```powershell
git remote add origin git@github.com:USERNAME/REPO.git
git push -u origin main
```

权限/认证说明：

- Windows PowerShell 会在需要时提示输入 GitHub 用户名/密码（对于 HTTPS），但现在 GitHub 推荐使用个人访问令牌（PAT）作为密码，或使用 `gh auth login` / SSH keys 以避免交互式密码提示。

附加建议（可选实现）

- 我可以为你：
  - A. 自动把修改的文件（`README.md` 和脚本）加入本地 git 并执行一次 commit（我可以在工作区运行这些命令，如果你允许我操作本机 git）；
  - B. 生成一份 `requirements.txt`（列出当前脚本可能需要的包）并加入仓库；
  - C. 帮你把仓库直接推送到 GitHub（需要你提供远程仓库 URL，且我需要在工作区能执行 git push，若你同意我可以运行这些命令）。

如果你同意让我在本机执行 git 操作（commit / push），请确认并提供：
- 你希望提交哪些文件（例如 `README.md` 以及那些修改过的脚本），以及
- 远程仓库 URL（或是否希望我先只本地 commit 而不 push）。

谢谢！

**仓库提交策略（只提交代码）**

- 本仓库默认只提交代码（`*.py`, `*.md`, `requirements.txt` 等文本/代码资源）。模型权重（如 `*.pt`、`*.pth`、`*.npy`）、缓存目录（`torch_cache/`）、`outputs/`、视频文件等会被加入 `.gitignore`，不随仓库提交。
- 我已经在仓库根添加了一个示例 `.gitignore`，包含常见的大文件模式（权重、缓存、checkpoints、视频等）。

如何仅提交代码（示例命令，在仓库根运行）：

```powershell
cd "e:/Documents/PolyU/CV/CVproject-CV"
# 确认 .gitignore 已添加
git status --ignored

# 只添加代码文件（示例：所有 Python 文件和 README）
git add -- "*.py" README.md

# 检查将要提交的内容
git status --porcelain

# 提交
git commit -m "Add code only: scripts and README"

# 连接远端（如尚未设置）并推送
git remote add origin https://github.com/M1chaelfmike/2025COMPCV.git
git branch -M main
git push -u origin main
```

如果仓库历史中已经包含大文件（例如你之前提交了模型权重），需要先从索引中移除这些被跟踪的大文件，示例（此操作不会删除本地文件，仅从 Git 跟踪中移除）：

```powershell
# 从索引移除所有已被跟踪的权重/二进制文件（保留本地副本）
git rm --cached "*.pt"
git rm --cached "*.pth"
git rm --cached "*.npy"
git rm -r --cached torch_cache || true

git commit -m "Remove large binaries from index (keep local copies); add .gitignore"
git push -u origin main
```

说明：如果你需要把权重也和仓库一起管理，推荐使用 Git LFS（需安装并注意 GitHub LFS 配额），或把权重放到 GitHub Releases /云盘，并在 README 中提供下载链接。我可以按需帮你配置 Git LFS 或把权重上传为 release。

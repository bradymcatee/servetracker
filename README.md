# servetracker

TrackNet-based tennis ball tracker and analyzer.
Detects and localizes the ball in video frames, maps positions to real-world court coordinates using a homography, and computes simple speed/trajectory statistics.

## Features
- Train a TrackNet-like model to predict ball heatmaps from 3-frame sequences.
- Run inference on videos to detect ball positions and save a visualization video.
- Compute homography from selected court points to map image positions to real-world court coordinates (meters).
- Estimate per-frame velocity and basic speed statistics (max / avg / std).
- Utilities for video preprocessing, visualization, and dataset handling.

## Requirements
- Python 3.8+
- CUDA (optional, for GPU training/inference)
- See `requirements.txt` for Python packages:
  - torch, torchvision, opencv-python, pandas, numpy, albumentations, tqdm, matplotlib

Install:
python -m pip install -r requirements.txt

## Repository layout (important files)
- src/
  - model.py            — TrackNetModel implementation (PyTorch)
  - train.py            — Training loop, checkpoints, visualization
  - inference.py        — Run model on videos, interactive court point selection, speed calculation
  - dataset.py          — TennisDataset loader (expects Dataset/game*/Clip*/Label.csv)
  - utils/
    - homography.py     — Homography helpers, velocity computation, filtering, visualization
    - video.py          — Video preprocessing (resize, read frames)
    - visualization.py  — Plotting & saving training/prediction visualizations
  - config.py           — Paths and default hyperparameters
- requirements.txt
- .gitignore

## Dataset format
The code expects a directory named `Dataset/` (or another path you provide) with this structure (as used by `TennisDataset`):
- Dataset/
  - game1/
    - Clip1/
      - <frame images> (e.g. frame0001.jpg)
      - Label.csv

Label.csv columns (the loader expects these names after loading and renaming):
- File_Name — image filename
- Visibility_Class — 1 if ball is visible
- X, Y — ball coordinates (dataset scaling in code assumes full-HD-ish original coords; dataset loader scales X/Y with 1280x720 reference)
- Trajectory_Pattern — (optional, used only as label column present)

Notes:
- The dataset code builds sequences of 3 consecutive frames where all three frames have Visibility_Class==1 and uses the middle frame as the target.
- The dataset loader normalizes/creates Gaussian target heatmaps.

## Quickstart — training
1. Prepare dataset and ensure structure above.
2. Create output directory for checkpoints/visuals (train script will create timestamps).
3. Example:
   python src/train.py --data_dir /path/to/Dataset --output_dir /path/to/outputs --num_workers 4

Important flags:
- --resume_from : path to checkpoint to resume.
- --checkpoint_freq : how often to save (epochs).

Training details:
- Model and training hyperparameters live in `src/config.py` (INPUT_HEIGHT, INPUT_WIDTH, BATCH_SIZE, etc.).
- Training saves checkpoints to outputs/training_{timestamp}/checkpoints and visualizations to .../visualizations.

## Quickstart — inference / processing a video
Example:
python src/inference.py \
  --video /path/to/video.mp4 \
  --model /path/to/model_best.pth \
  --output /path/to/output_video.mp4 \
  --device cuda \
  --visualize-homography \
  --overlay-court

Notes:
- The first frame will open an interactive window to select court points. The script requires at least the first 6 points (near/far service line left/right/center) and supports selecting up to 12 points to improve accuracy.
- After homography is computed the script will process frames, draw trajectories, compute speeds and save an output video and summary speeds printed to console.
- If CUDA isn't available, pass `--device cpu`.

## Utilities
- Video preprocessing: `src/utils/video.py::preprocess_video(video_path, output_path=None, target_fps=30)` — reads and resizes frames; returns list of frames.
- Homography and speeds: `src/utils/homography.py` — helpers to compute/refine homography, map points, compute velocity, and smooth speeds.
- Visualization: `src/utils/visualization.py` — saving training plots, sample predictions, and trajectory overlays.

## Tips & gotchas
- The dataset loader expects coordinates scaled to a 1280x720 reference when converting to heatmap indices; verify your Label.csv X/Y units match expectations.
- The homography workflow is interactive; selecting accurate court points is critical for good real-world speed estimates.
- The model outputs a per-pixel heatmap (softmax over channels), and training uses focal-BCE style loss in train.py.
- Default learning rate in config.py is 1.0 because training uses Adadelta; tweak if you change optimizer.

## Contact / Author
Repository: https://github.com/bradymcatee/servetracker
Author: @bradymcatee

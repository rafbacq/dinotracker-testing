# DINO-Tracker for Liver Ultrasound Videos

Automated pipeline for applying [DINO-Tracker](https://dino-tracker.github.io/) (ECCV 2024) to grayscale liver ultrasound video sequences.

## How DINO-Tracker Works

DINO-Tracker performs **per-video self-supervised point tracking**. Unlike pretrained tracking models, it trains a model **from scratch on each individual video** using:

1. **RAFT optical flow** → creates pseudo ground-truth trajectories
2. **DINOv2 best-buddy correspondences** → contrastive learning signal for feature refinement
3. **DeltaDINO module** → learns video-specific residual features on top of DINOv2
4. **TrackerHead CNN** → predicts precise point positions from correlation maps

This means every video gets its own trained model — there are no "pretrained weights."

## Setup

```bash
# Create environment
conda create -n dino-tracker python=3.9
conda activate dino-tracker
pip install -r requirements.txt

# Add project to PYTHONPATH
export PYTHONPATH=$(pwd):$PYTHONPATH
```

> **Note**: A CUDA-capable GPU is required for training and preprocessing (RAFT, DINOv2).

## Google Colab

**Option A** — Use the provided notebook:
Upload and open `run_on_colab.ipynb` in Google Colab (select T4 GPU runtime).

**Option B** — Manual terminal setup:
```bash
# In Colab terminal:
git clone https://github.com/rafbacq/dinotracker-testing.git /content/dinotracker-testing
cd /content/dinotracker-testing
bash scripts/setup_colab.sh            # Fixes numpy/OpenCV version conflict
export PYTHONPATH=$(pwd):$PYTHONPATH
bash scripts/run_all.sh
```

## Quick Start

### Process all videos
```bash
bash scripts/run_all.sh
```

### Process a single video
```bash
bash scripts/run_all.sh volunteer01.mp4
```

### Step-by-step (per video)
```bash
# 1. Preprocess (extract frames + optical flow + DINO features)
bash scripts/preprocess_video.sh volunteer01

# 2. Train (self-supervised, ~8000 iterations)
bash scripts/train_video.sh volunteer01

# 3. Inference + Visualization
bash scripts/infer_video.sh volunteer01
```

### Python orchestrator (full control)
```bash
python run_ultrasound_pipeline.py \
    --input-dir usliverseq-mp4 \
    --output-dir outputs \
    --target-frames 150 \
    --vis-fps 10

# Skip already-completed steps:
python run_ultrasound_pipeline.py --skip-extraction --skip-preprocessing

# Process specific videos:
python run_ultrasound_pipeline.py --videos volunteer01.mp4 volunteer03.mp4
```

## Output Structure

```
outputs/
├── volunteer01/
│   ├── video/                    # Extracted RGB frames (grayscale triplicated)
│   ├── of_trajectories/          # RAFT optical flow trajectories
│   │   ├── trajectories.pt
│   │   ├── fg_trajectories.pt    # Foreground-only trajectories
│   │   └── bg_trajectories.pt    # Background-only trajectories
│   ├── dino_embeddings/          # DINOv2 feature maps per frame
│   ├── dino_best_buddies/        # DINO best-buddy correspondences
│   ├── masks/                    # Auto-generated foreground masks
│   ├── models/dino_tracker/      # Trained checkpoints
│   │   ├── delta_dino_*.pt
│   │   └── tracker_head_*.pt
│   ├── grid_trajectories/        # Predicted trajectories (NPY)
│   ├── grid_occlusions/          # Predicted occlusions (NPY)
│   └── visualizations/           # Output MP4 with tracked points
├── volunteer02/
│   └── ...
├── logs/                         # Pipeline execution logs
└── pipeline_summary.json         # Structured results JSON
```

## Design Choices

### Grayscale → RGB Conversion
DINOv2 expects 3-channel RGB input with ImageNet normalization. We **triplicate** the grayscale channel (stack it 3 times) rather than apply a colormap. This preserves the original intensity structure without injecting artificial color patterns.

### Temporal Subsampling
Ultrasound videos are 5–10 minutes at 14–25 fps (4,200–15,000 frames). DINO-Tracker was designed for short natural videos (~50–200 frames). We subsample to **~150 frames** per video, computed automatically based on video length. This preserves the overall motion pattern while keeping computation tractable.

### Resolution
Ultrasound frames have roughly 4:3 aspect ratio, unlike the 16:9 used in the original DINO-Tracker demos. We resize to **420×560** to match the aspect ratio while staying close to the original model's expected input size.

## Limitations

1. **Domain gap**: DINOv2 features were learned on natural RGB images. On triplicated grayscale ultrasound, features have reduced discriminative power. DINO best-buddy correspondences may be noisier, potentially affecting training convergence.

2. **Speckle noise**: Ultrasound images contain multiplicative speckle noise. RAFT optical flow may produce noisier trajectories compared to natural video, since it was not trained on ultrasound data.

3. **Low contrast**: Liver ultrasound has subtle intensity gradients compared to natural video. The foreground/background segmentation masks (computed via DINO PCA saliency) may not align with anatomically meaningful regions.

4. **Temporal subsampling**: Subsampling from ~10,000 to ~150 frames loses fine-grained temporal dynamics. Rapid breathing phases may be underrepresented.

5. **No ground truth**: Without ground-truth anatomical landmarks, there is no quantitative evaluation metric. Results are qualitative only.

## Suggestions for Future Improvement

1. **Fine-tune RAFT on ultrasound**: Train or fine-tune RAFT on ultrasound optical flow datasets for better trajectory quality.
2. **Ultrasound-specific DINO**: Consider a DINOv2 model fine-tuned on medical images (e.g., from the BiomedCLIP or RAD-DINO family).
3. **Sliding window**: Instead of subsampling the full video, process overlapping temporal windows and stitch trajectories.
4. **Speckle filtering**: Apply speckle-reduction filters (e.g., non-local means, Lee filter) before frame extraction.
5. **Landmark evaluation**: If anatomical landmarks are available, implement quantitative accuracy metrics.
6. **Multi-scale tracking**: Use multiple grid intervals to capture both coarse organ motion and fine tissue deformation.

## Dataset Information

| Volunteer | Spatial Res (mm/px) | FPS | Frequency (MHz) |
|-----------|-------------------|-----|-----------------|
| 01 | 0.71 | 25 | 2.22 |
| 02 | 0.40 | 16 | 2.00 |
| 03 | 0.36 | 17 | 1.82 |
| 04 | 0.42 | 15 | 2.22 |
| 05 | 0.40 | 15 | 2.22 |
| 06 | 0.37 | 17 | 1.82 |
| 07 | 0.28 | 14 | 2.22 |

Source: Antares ultrasound (Siemens), CH4-1 transducer, Geneva University Hospital.

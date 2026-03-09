#!/bin/bash
# ============================================================================
# Run inference + visualization for a single video (training must be done first)
# Usage: bash scripts/infer_video.sh <video_name>
#   e.g. bash scripts/infer_video.sh volunteer01
# ============================================================================

set -e

if [ -z "$1" ]; then
    echo "Usage: bash scripts/infer_video.sh <video_name>"
    echo "  e.g. bash scripts/infer_video.sh volunteer01"
    exit 1
fi

VIDEO_NAME="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

DATA_PATH="outputs/$VIDEO_NAME"

echo "=== Step 1/2: Grid inference for $VIDEO_NAME ==="
python inference_grid.py \
    --config config/ultrasound_train.yaml \
    --data-path "$DATA_PATH" \
    --interval 10 \
    --use-segm-mask

echo "=== Step 2/2: Visualizing trajectories for $VIDEO_NAME ==="
python visualization/visualize_rainbow.py \
    --data-path "$DATA_PATH" \
    --infer-res-size 280 378 \
    --of-res-size 280 378 \
    --fps 10

echo "=== Inference + visualization complete for $VIDEO_NAME ==="
echo "  Trajectories: $DATA_PATH/grid_trajectories/"
echo "  Visualizations: $DATA_PATH/visualizations/"

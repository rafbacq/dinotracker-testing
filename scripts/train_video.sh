#!/bin/bash
# ============================================================================
# Run training only for a single video (preprocessing must be done first)
# Usage: bash scripts/train_video.sh <video_name>
#   e.g. bash scripts/train_video.sh volunteer01
# ============================================================================

set -e

if [ -z "$1" ]; then
    echo "Usage: bash scripts/train_video.sh <video_name>"
    echo "  e.g. bash scripts/train_video.sh volunteer01"
    exit 1
fi

VIDEO_NAME="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

DATA_PATH="outputs/$VIDEO_NAME"

echo "=== Training DINO-Tracker on $VIDEO_NAME ==="
python train.py \
    --config config/ultrasound_train.yaml \
    --data-path "$DATA_PATH"

echo "=== Training complete for $VIDEO_NAME ==="
echo "  Checkpoints saved in $DATA_PATH/models/dino_tracker/"

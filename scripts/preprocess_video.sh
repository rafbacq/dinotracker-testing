#!/bin/bash
# ============================================================================
# Run preprocessing only for a single video
# Usage: bash scripts/preprocess_video.sh <video_name>
#   e.g. bash scripts/preprocess_video.sh volunteer01
# ============================================================================

set -e

if [ -z "$1" ]; then
    echo "Usage: bash scripts/preprocess_video.sh <video_name>"
    echo "  e.g. bash scripts/preprocess_video.sh volunteer01"
    exit 1
fi

VIDEO_NAME="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

DATA_PATH="outputs/$VIDEO_NAME"
MP4_PATH="usliverseq-mp4/${VIDEO_NAME}.mp4"

echo "=== Step 1/2: Extracting frames from $MP4_PATH ==="
python preprocessing/extract_ultrasound_frames.py \
    --video-path "$MP4_PATH" \
    --output-folder "$DATA_PATH/video" \
    --target-frames 150

echo "=== Step 2/2: Running preprocessing pipeline ==="
python preprocessing/main_preprocessing.py \
    --config config/ultrasound_preprocessing.yaml \
    --data-path "$DATA_PATH"

echo "=== Preprocessing complete for $VIDEO_NAME ==="

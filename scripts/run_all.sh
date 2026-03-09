#!/bin/bash
# ============================================================================
# DINO-Tracker Ultrasound Pipeline — Master Runner
#
# Processes all liver ultrasound videos through the complete DINO-Tracker
# pipeline: frame extraction → preprocessing → training → inference → viz.
#
# Usage:
#   bash scripts/run_all.sh                    # Process all videos
#   bash scripts/run_all.sh volunteer01.mp4    # Process one video
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Ensure PYTHONPATH includes project root
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo "=============================================="
echo "  DINO-Tracker Ultrasound Pipeline"
echo "  Project root: $PROJECT_ROOT"
echo "=============================================="

# Parse optional video arguments
VIDEO_ARGS=""
if [ $# -gt 0 ]; then
    VIDEO_ARGS="--videos $*"
    echo "  Processing specific videos: $*"
fi

# Run the orchestrator
python run_ultrasound_pipeline.py \
    --input-dir usliverseq-mp4 \
    --output-dir outputs \
    --preprocess-config config/ultrasound_preprocessing.yaml \
    --train-config config/ultrasound_train.yaml \
    --target-frames 150 \
    --vis-fps 10 \
    --grid-interval 10 \
    $VIDEO_ARGS

echo ""
echo "Pipeline complete! Check outputs/ for results."
echo "Logs are in outputs/logs/"

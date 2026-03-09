#!/bin/bash
# ============================================================================
# Colab-specific setup for DINO-Tracker Ultrasound Pipeline
#
# Run this FIRST in a Colab terminal/cell before running the pipeline:
#   bash scripts/setup_colab.sh
#
# This fixes version incompatibilities between Colab's default packages
# and the DINO-Tracker requirements.
# ============================================================================

set -e

echo "=== DINO-Tracker Colab Setup ==="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# 1. Fix numpy/OpenCV incompatibility
#    Colab ships numpy 2.x but old opencv-python needs numpy 1.x.
#    Solution: install compatible versions of both.
echo ">>> Installing compatible numpy and OpenCV..."
pip install --quiet "numpy<2" "opencv-python>=4.9"

# 2. Install PyTorch (Colab may already have it, but ensure correct version)
echo ">>> Ensuring PyTorch with CUDA..."
pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu121 2>/dev/null || \
    echo "  (PyTorch already installed via Colab)"

# 3. Install remaining dependencies (skip version-pinned torch/torchvision)
echo ">>> Installing remaining dependencies..."
pip install --quiet einops imageio imageio-ffmpeg kornia matplotlib mediapy \
    pandas pillow tqdm PyYAML antialiased_cnns xformers wandb 2>/dev/null || \
    echo "  (Some packages may have warnings, this is usually OK)"

# 4. Set PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# 5. Verify critical imports
echo ">>> Verifying imports..."
python3 -c "
import numpy; print(f'  numpy:      {numpy.__version__}')
import cv2;   print(f'  opencv:     {cv2.__version__}')
import torch;  print(f'  torch:      {torch.__version__}')
print(f'  CUDA avail: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:        {torch.cuda.get_device_name(0)}')
import yaml;  print(f'  pyyaml:     OK')
import einops; print(f'  einops:     OK')
import kornia; print(f'  kornia:     OK')
print('All imports successful!')
"

echo ""
echo "=== Setup complete! ==="
echo "Now run:  export PYTHONPATH=$(pwd):\$PYTHONPATH && bash scripts/run_all.sh"

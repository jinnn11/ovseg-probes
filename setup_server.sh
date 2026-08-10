#!/usr/bin/env bash
# Setup GPU server environment for ovseg-probes inference.
# Run once after SSH-ing into the instance.
set -euo pipefail

REPO_URL="${REPO_URL:-git@github.com:YOUR_USER/ovseg-probes.git}"
PYTHON_BIN="${PYTHON:-python3}"
WEIGHTS_DIR="weights"

echo "=== 1. Clone repo ==="
if [ ! -d "ovseg-probes" ]; then
    git clone "$REPO_URL"
fi
cd ovseg-probes

echo "=== 2. Python environment ==="
"${PYTHON_BIN}" -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

echo "=== 3. Install dependencies ==="
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Grounded-SAM dependencies
pip install groundingdino-py segment-anything opencv-python supervision

echo "=== 4. Download model weights ==="
mkdir -p "$WEIGHTS_DIR"

# Grounding DINO Swin-T
GDINO_URL="https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
if [ ! -f "$WEIGHTS_DIR/groundingdino_swint_ogc.pth" ]; then
    echo "Downloading Grounding DINO Swin-T..."
    wget -q -P "$WEIGHTS_DIR" "$GDINO_URL"
fi

# SAM ViT-L
SAM_URL="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
if [ ! -f "$WEIGHTS_DIR/sam_vit_l_0b3195.pth" ]; then
    echo "Downloading SAM ViT-L..."
    wget -q -P "$WEIGHTS_DIR" "$SAM_URL"
fi

echo "=== 5. Download probe images ==="
python -m src.download_images --workers 16

echo "=== 6. Run tests ==="
python -m pytest -x -q

echo ""
echo "=== Setup complete ==="
echo "Weights: $WEIGHTS_DIR/"
echo "Images:  data/images/"
echo "Next:    bash run_all.sh"

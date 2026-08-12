#!/usr/bin/env bash
# One-shot server setup for ovseg-probes on a fresh GPU instance (Vast.ai, Lambda, etc.).
#
# Usage (on the server):
#   git clone https://github.com/jinnn11/ovseg-probes.git
#   cd ovseg-probes
#   bash setup_server.sh
#
# What this does:
#   1. Creates a venv and installs all Python deps
#   2. Installs Grounding DINO + SAM model weights
#   3. Downloads probe images from COCO/LVIS/VG
#   4. Prepares RefCOCO validation data (HuggingFace, not the dead UNC server)
#   5. Runs a 5-probe smoke test to verify GPU inference works
#
# After setup, run the full pipeline:
#   bash run_all.sh
#
# Then pull results to your Mac:
#   bash pull_results.sh root@<server-ip> <ssh-port> /workspace/ovseg-probes
set -euo pipefail

WORK_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$WORK_DIR"

echo "=========================================="
echo "  ovseg-probes server setup"
echo "  Working directory: $WORK_DIR"
echo "=========================================="

# ── Step 1: Python environment ───────────────────────────────────

echo ""
echo "=== Step 1/6: Python environment ==="

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "Created .venv"
else
    echo ".venv already exists, reusing"
fi

source .venv/bin/activate

pip install --upgrade pip -q

# Core deps
pip install -q \
    torch torchvision \
    pycocotools lvis numpy matplotlib Pillow \
    requests tqdm opencv-python \
    datasets  # for RefCOCO via HuggingFace

echo "Python deps installed."

# ── Step 2: Grounding DINO ───────────────────────────────────────

echo ""
echo "=== Step 2/6: Grounding DINO ==="

GDINO_DIR="third_party/GroundingDINO"
if [ ! -d "$GDINO_DIR" ]; then
    mkdir -p third_party
    git clone https://github.com/IDEA-Research/GroundingDINO.git "$GDINO_DIR"
    cd "$GDINO_DIR"
    pip install -e . -q
    cd "$WORK_DIR"
    echo "Grounding DINO installed."
else
    echo "Grounding DINO already cloned."
fi

GDINO_WEIGHTS="weights/groundingdino_swint_ogc.pth"
if [ ! -f "$GDINO_WEIGHTS" ]; then
    mkdir -p weights
    echo "Downloading Grounding DINO weights (Swin-T)..."
    wget -q --show-progress -O "$GDINO_WEIGHTS" \
        "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
else
    echo "Grounding DINO weights already present."
fi

# ── Step 3: SAM ──────────────────────────────────────────────────

echo ""
echo "=== Step 3/6: SAM ==="

pip install -q segment-anything 2>/dev/null || pip install -q git+https://github.com/facebookresearch/segment-anything.git

SAM_WEIGHTS="weights/sam_vit_l_0b3195.pth"
if [ ! -f "$SAM_WEIGHTS" ]; then
    mkdir -p weights
    echo "Downloading SAM weights (ViT-L)..."
    wget -q --show-progress -O "$SAM_WEIGHTS" \
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
else
    echo "SAM weights already present."
fi

# ── Step 4: Probe images ─────────────────────────────────────────

echo ""
echo "=== Step 4/6: Probe images ==="

if [ -d "data/images" ] && [ "$(find data/images -name '*.jpg' 2>/dev/null | wc -l)" -gt 100 ]; then
    echo "Probe images already downloaded ($(find data/images -name '*.jpg' | wc -l) images)."
else
    echo "Downloading probe images..."
    python -m src.download_images
fi

# ── Step 5: RefCOCO validation data ──────────────────────────────

echo ""
echo "=== Step 5/6: RefCOCO validation data ==="
# The original UNC server (bvisionweb1.cs.unc.edu) is frequently down.
# We use lmms-lab/RefCOCO from HuggingFace instead.

if [ -f "data/refcoco/refcoco_val.json" ]; then
    echo "RefCOCO JSON already exists."
else
    echo "Converting RefCOCO from HuggingFace..."
    mkdir -p data/refcoco
    python scripts/make_refcoco_json.py
fi

REFCOCO_IMGS="$(find data/images/coco -name '*.jpg' 2>/dev/null | wc -l)"
if [ "$REFCOCO_IMGS" -gt 150 ]; then
    echo "RefCOCO images already downloaded ($REFCOCO_IMGS images)."
else
    echo "Downloading COCO images for RefCOCO eval..."
    python scripts/download_refcoco_images.py
fi

# ── Step 6: Smoke test ───────────────────────────────────────────

echo ""
echo "=== Step 6/6: Smoke test (5 probes) ==="
echo "Running 5-probe inference to verify GPU setup..."

bash run_all.sh --smoke 5

echo ""
echo "=========================================="
echo "  Setup complete."
echo ""
echo "  Next steps:"
echo "    1. Run full pipeline:  bash run_all.sh"
echo "    2. Pull to Mac:        bash pull_results.sh root@<ip> <port> $WORK_DIR"
echo "=========================================="

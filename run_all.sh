#!/usr/bin/env bash
# Run full inference + analysis pipeline. Tmux-friendly (one shot).
# Usage: bash run_all.sh [model_name]
#   model_name: mock | mock_cheat | grounded_sam (default: grounded_sam)
set -euo pipefail

MODEL="${1:-grounded_sam}"
PROBE_FILE="probes/probe_set_v1.json"
CONTROL_FILE="probes/control_set_v1.json"
PRED_DIR="predictions"
RESULTS_DIR="results/${MODEL}"

source .venv/bin/activate

echo "========================================"
echo "  Model: ${MODEL}"
echo "  Probes: ${PROBE_FILE}"
echo "  Controls: ${CONTROL_FILE}"
echo "========================================"

echo ""
echo "=== Step 1: Inference on distractor probes ==="
python -m src.run_inference \
    --model "$MODEL" \
    --probe-file "$PROBE_FILE" \
    --out-dir "$PRED_DIR"

echo ""
echo "=== Step 2: Inference on control probes ==="
python -m src.run_inference \
    --model "$MODEL" \
    --probe-file "$CONTROL_FILE" \
    --out-dir "$PRED_DIR"

echo ""
echo "=== Step 3: Oracle-box inference (SAM with GT boxes) ==="
python -m src.run_inference \
    --model "$MODEL" \
    --probe-file "$PROBE_FILE" \
    --out-dir "$PRED_DIR" \
    --oracle-box

python -m src.run_inference \
    --model "$MODEL" \
    --probe-file "$CONTROL_FILE" \
    --out-dir "$PRED_DIR" \
    --oracle-box

echo ""
echo "=== Step 4: Analysis ==="
python -m src.analyze \
    --predictions-dir "${PRED_DIR}/${MODEL}" \
    --probes "$PROBE_FILE" \
    --controls "$CONTROL_FILE" \
    --output-dir "$RESULTS_DIR"

echo ""
echo "=== Step 5: Oracle analysis ==="
python -m src.analyze \
    --predictions-dir "${PRED_DIR}/${MODEL}_oracle" \
    --probes "$PROBE_FILE" \
    --controls "$CONTROL_FILE" \
    --output-dir "${RESULTS_DIR}_oracle"

echo ""
echo "========================================"
echo "  Done. Results in: ${RESULTS_DIR}/"
echo "  Pull with: bash pull_results.sh"
echo "========================================"

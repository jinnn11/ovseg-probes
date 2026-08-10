#!/usr/bin/env bash
# Run full inference + analysis pipeline. Tmux-friendly (one shot).
# Usage:
#   bash run_all.sh                        # Full run with RefCOCO gate
#   bash run_all.sh --smoke 5              # Smoke test: 5 probes only
#   bash run_all.sh --skip-refcoco         # Skip RefCOCO gate
#   bash run_all.sh mock                   # Use mock model (local testing)
set -euo pipefail

SKIP_REFCOCO=false
MODEL="grounded_sam"
SMOKE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-refcoco) SKIP_REFCOCO=true; shift ;;
        --smoke) SMOKE="$2"; SKIP_REFCOCO=true; shift 2 ;;
        mock|mock_cheat|grounded_sam) MODEL="$1"; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

MAX_PROBES_FLAG=""
if [ "$SMOKE" -gt 0 ]; then
    MAX_PROBES_FLAG="--max-probes $SMOKE"
    echo "*** SMOKE TEST: $SMOKE probes per step ***"
fi

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

# ── Session 1: RefCOCO sanity check ──────────────────────────────

if [ "$SKIP_REFCOCO" = false ] && [ -d "data/refcoco" ]; then
    echo ""
    echo "=== SESSION 1: RefCOCO sanity check (500 samples) ==="
    python -m src.eval_refcoco \
        --model "$MODEL" \
        --refcoco-dir data/refcoco \
        --split val \
        --max-samples 500 \
        --output "results/refcoco_${MODEL}.json"

    # Gate: check accuracy > 50% before proceeding
    ACC=$(python -c "
import json
with open('results/refcoco_${MODEL}.json') as f:
    r = json.load(f)
print(f\"{r['accuracy']:.3f}\")
")
    echo "RefCOCO accuracy: ${ACC}"
    if python -c "exit(0 if float('${ACC}') > 0.50 else 1)"; then
        echo "PASS — proceeding to probe inference"
    else
        echo "FAIL — accuracy below 50%. Check model setup before continuing."
        echo "To skip this gate: bash run_all.sh --skip-refcoco"
        exit 1
    fi
elif [ "$SKIP_REFCOCO" = false ]; then
    echo ""
    echo "RefCOCO data not found in data/refcoco/, skipping sanity check."
    echo "To set up: see setup_server.sh step 6"
fi

# ── Session 2: Full probe inference ──────────────────────────────

echo ""
echo "=== SESSION 2, Step 1: Inference on distractor probes ==="
python -m src.run_inference \
    --model "$MODEL" \
    --probe-file "$PROBE_FILE" \
    --out-dir "$PRED_DIR" \
    $MAX_PROBES_FLAG

echo ""
echo "=== SESSION 2, Step 2: Inference on control probes ==="
python -m src.run_inference \
    --model "$MODEL" \
    --probe-file "$CONTROL_FILE" \
    --out-dir "$PRED_DIR" \
    $MAX_PROBES_FLAG

echo ""
echo "=== SESSION 2, Step 3: Oracle-box inference (SAM with GT boxes) ==="
python -m src.run_inference \
    --model "$MODEL" \
    --probe-file "$PROBE_FILE" \
    --out-dir "$PRED_DIR" \
    $MAX_PROBES_FLAG \
    --oracle-box

python -m src.run_inference \
    --model "$MODEL" \
    --probe-file "$CONTROL_FILE" \
    --out-dir "$PRED_DIR" \
    $MAX_PROBES_FLAG \
    --oracle-box

echo ""
echo "=== SESSION 2, Step 4: Analysis ==="
python -m src.analyze \
    --predictions-dir "${PRED_DIR}/${MODEL}" \
    --probes "$PROBE_FILE" \
    --controls "$CONTROL_FILE" \
    --output-dir "$RESULTS_DIR"

echo ""
echo "=== SESSION 2, Step 5: Oracle analysis ==="
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

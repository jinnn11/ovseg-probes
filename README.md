# ovseg-probes

Diagnostic probe set and evaluation pipeline for testing compositional grounding in open-vocabulary segmentation models. Evaluates whether models like Grounding DINO + SAM actually compose multi-word phrases (negation, spatial relations, attribute binding, fine-grained discrimination) or just match nouns.

**Paper:** *Does Open-Vocabulary Segmentation Actually Read the Prompt? A Controlled Probe Study of Compositional Grounding and Failure Attribution*

## Key Results

| Phenomenon | Distractor Acc | Control Acc | Gap |
|---|---|---|---|
| attribute_color | 74.9% | 92.0% | 17.1% |
| finegrained_confusable | 85.5% | 88.8% | 3.2% |
| negation | **20.7%** | 93.2% | **72.5%** |
| spatial | **~29%** | **~80%** | **~51%** |

- Negation accuracy is below the ~50% chance baseline (model does the opposite of "not")
- Spatial pair consistency: **0/51** (model picks a preferred instance regardless of left/right)
- Oracle condition (SAM with GT boxes): **99.6%**, pinning all failures to the grounding stage
- Fine-grained gap of 3.2% confirms probes don't manufacture failures where capability exists

## Probe Set

914 human-verified probes across 608 images:

| Phenomenon | Distractor | Control | Source |
|---|---|---|---|
| attribute_color | 199 | 50 | Visual Genome |
| spatial_left | 78 | 35 | COCO |
| spatial_right | 72 | 37 | COCO |
| finegrained_confusable | 69 | 80 | LVIS |
| finegrained_distinct | 48 | -- | LVIS |
| negation | 82 | 59 | Visual Genome |
| negation_positive | 105 | -- | Visual Genome |
| **Total** | **653** | **261** | |

238 complete mirror pairs, 177 singletons. 48% keep rate from 1,910 candidates. Inter-annotator agreement: 86.4% (165/191).

## Quick Start

### On a GPU server (Vast.ai, Lambda, etc.)

```bash
git clone https://github.com/jinnn11/ovseg-probes.git
cd ovseg-probes
bash setup_server.sh     # installs everything, downloads data, runs smoke test
bash run_all.sh          # full 914-probe inference + RefCOCO gate + oracle
```

`setup_server.sh` handles: venv creation, pip deps, Grounding DINO + SAM weights, probe image downloads, RefCOCO validation data (via HuggingFace, not the frequently-down UNC server), and a 5-probe smoke test.

### Pull results to your Mac

```bash
bash pull_results.sh root@<server-ip> <ssh-port> /workspace/ovseg-probes
```

### Local development (no GPU needed)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest                    # run tests
bash run_all.sh mock                # mock model, no GPU
```

## Repository Layout

```
probes/                   # frozen probe JSONs (committed)
  probe_set_v1.json       # 653 distractor probes
  control_set_v1.json     # 261 control probes
src/
  mine_spatial.py          # spatial-relations miner (COCO)
  mine_attributes.py       # attribute-binding miner (Visual Genome)
  mine_finegrained.py      # fine-grained miner (LVIS)
  mine_negation.py         # negation miner (Visual Genome)
  download_images.py       # concurrent image downloader
  freeze.py                # compile verified probes into frozen set
  run_inference.py         # per-probe inference with checkpointing
  grounded_sam.py          # Grounding DINO + SAM wrapper
  analyze.py               # accuracy, gap, pairs, failures, sweep
  eval_refcoco.py          # RefCOCO validation gate
  gallery.py               # failure gallery renderer
  appendix_analysis.py     # per-color, per-item, per-group breakdowns
  schema.py                # Probe dataclass
  geometry.py              # IoU, box conversion
  mock_model.py            # mock models for local testing
scripts/
  make_refcoco_json.py     # convert HuggingFace RefCOCO to eval format
  download_refcoco_images.py  # download COCO images for RefCOCO eval
notebooks/                 # verification notebooks (verify.ipynb, reverify.ipynb)
data/                      # local annotations and images (gitignored)
predictions/               # model outputs (gitignored)
results/                   # analysis outputs (gitignored)
gallery/                   # failure visualizations (gitignored)
report/                    # LaTeX report and figures (gitignored)
```

## Pipeline

### 1. Mining

Each miner extracts probe candidates from dataset annotations:

```bash
python -m src.mine_spatial         # COCO train2017 -> spatial probes
python -m src.mine_attributes      # VG attributes -> color-binding probes
python -m src.mine_finegrained     # LVIS v1 train -> sibling-category probes
python -m src.mine_negation        # VG relationships -> negation probes
```

### 2. Verification

Interactive notebook (`verify.ipynb`) for human review. Each probe is accepted or rejected based on: image matches prompt, boxes are correctly assigned, compositional contrast is unambiguous. Decisions saved to `decisions.jsonl`.

### 3. Freeze

```bash
python -m src.freeze
```

Compiles verified keepers into `probes/probe_set_v1.json` and `probes/control_set_v1.json`.

### 4. Inference

```bash
bash run_all.sh                    # full pipeline
bash run_all.sh --smoke 5          # smoke test (5 probes)
bash run_all.sh --skip-refcoco     # skip RefCOCO gate
```

The pipeline runs:
1. RefCOCO validation gate (500 samples, expects ~51% for Swin-T zero-shot)
2. Distractor probe inference
3. Control probe inference
4. Oracle-box inference (SAM with GT boxes)
5. Analysis (accuracy, gap, pairs, failures, threshold sweep)

### 5. Analysis outputs

Each run produces in `results/{model}/`:

| File | Contents |
|---|---|
| `report.json` | full structured report |
| `accuracy.csv` | per-phenomenon accuracy |
| `distractor_gap.csv` | control - distractor accuracy |
| `pair_consistency.csv` | mirror-pair both-correct rates |
| `failures.csv` | per-probe failure classification |
| `threshold_sweep.csv` | accuracy at confidence cutoffs 0.15-0.50 |
| `accuracy_bars.png` | accuracy bar chart |
| `failure_bars.png` | failure classification chart |
| `distractor_gap.png` | gap visualization |
| `threshold_sweep.png` | threshold sweep plot |

### 6. Gallery

```bash
python -m src.gallery              # render failure images
python -m src.appendix_analysis    # per-color, per-item breakdowns
```

## Coordinate Convention

All boxes in this project use `xyxy` pixel coordinates:

```
[x_min, y_min, x_max, y_max]
```

COCO annotations use `xywh` and are converted to `xyxy` exactly once at load time. Do not mix coordinate systems inside project files.

## Prediction Format

Per probe: `predictions/{model}/{probe_id}.json`

```json
{
  "probe_id": "spatial_0012",
  "model": "detection",
  "candidates": [{"box_xyxy": [...], "confidence": 0.85, "mask_rle": {...}}],
  "top_box": [...],
  "top_confidence": 0.85,
  "top_mask": {...},
  "wall_time_s": 0.123
}
```

## Hardware

All inference was run on a single NVIDIA RTX 3090 (24 GB VRAM) via Vast.ai. Full pipeline (914 probes + 500 RefCOCO + oracle) completes in a single session.

## Scoring

A prediction is **correct** if:
- IoU >= 0.5 with the target box, AND
- IoU < 0.5 with the distractor box (when present)

Controls only require IoU >= 0.5 with the target (no distractor).

**Distractor gap** = control accuracy - distractor accuracy (isolates the compositional cost).

**Pair consistency** = fraction of intact mirror pairs where both mirrors are correct. Only computed on pairs where both probes survived verification.

## Mining Details

See `mining_log.md` for full mining parameters, thresholds, yields, and per-phenomenon design decisions.

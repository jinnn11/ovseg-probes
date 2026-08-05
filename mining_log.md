# Mining Log

## Spatial-Relations Miner

**Date:** 2026-08-05
**Script:** `src/mine_spatial.py`
**Source:** `data/coco/instances_train2017.json`

### Thresholds

| Parameter | Value |
|---|---|
| MIN_HORIZ_GAP_FRAC | 0.30 (A centers > 30% image width apart) |
| MAX_VERT_DIFF_FRAC | 0.10 (A centers < 10% image height apart) |
| MIN_BOX_AREA_FRAC | 0.03 (each box > 3% image area) |
| MAX_BOX_AREA_FRAC | 0.25 (each box < 25% image area) |
| MAX_DISTRACTOR_PAIRS | 190 (subsampled from full set) |
| MAX_CONTROL_PROBES | 100 (50 left + 50 right) |

### Allowed Categories (27)

bicycle, car, motorcycle, bus, truck, dog, cat, horse, cow, sheep,
elephant, bear, backpack, umbrella, handbag, suitcase, bottle, cup, bowl,
chair, couch, potted plant, laptop, tv, clock, vase, teddy bear

### Yields (before subsampling)

- Indexed: 294,394 annotations across 71,429 images (allowed categories only)
- Raw distractor pairs: 312 (624 probes from 528 images)
- Raw control probes: ~3,746

### Yields (after subsampling)

| File | Probes | Images | Pairs | Left | Right |
|---|---|---|---|---|---|
| `spatial_distractor.json` | 380 | 174 | 190 | 190 | 190 |
| `spatial_control.json` | 100 | 99 | — | 50 | 50 |
| `spatial_extras.json` | 244 | 117 | 122 | 122 | 122 |

### Notes

- All boxes stored as xyxy pixels; COCO xywh converted once at load time.
- Mirrors linked via `pair_id` (e.g. `sp_0001`) for pair-consistency analysis.
- Distractor set subsampled by pair_id (mirrors always kept together, seed=42).
- Controls subsampled stratified left/right (seed=42).
- Extras saved separately in case verification rejection rate is high.
- Crowd annotations (`iscrowd=1`) excluded.

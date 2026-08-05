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

---

## Fine-Grained Miner

**Date:** 2026-08-05
**Script:** `src/mine_finegrained.py`
**Source:** `data/lvis/lvis_v1_train.json`

### Sibling Groups (10)

Each group tagged as **confusable** (coarse model would collapse into one
concept) or **distinct** (clearly different objects — easy-tier baseline).

| Group | Tier | Categories |
|---|---|---|
| bags | confusable | handbag, backpack, suitcase, tote bag, shoulder bag |
| cup_mug_glass | confusable | cup, mug, drinking glass, wine glass |
| bottles | confusable | wine bottle, water bottle, beer bottle |
| utensils | confusable | spatula, spoon, tongs, ladle, wooden spoon |
| cooking_vessels | confusable | pot, frying pan, pan, wok, kettle |
| pillow_cushion | confusable | pillow, cushion |
| bowl_plate | distinct | bowl, plate, saucer, platter |
| knife_scissors | distinct | knife, scissors |
| headwear | distinct | hat, helmet, beanie, cap |
| seating | distinct | armchair, stool, chair |

### Thresholds

| Parameter | Value |
|---|---|
| MIN_BOX_AREA_FRAC | 0.01 (each box > 1% image area) |
| MAX_BOX_AREA_FRAC | 0.50 (each box < 50% image area) |
| MAX_GROUP_FRAC | 0.25 (cap any group at 25% of distractor total) |
| MAX_CONTROL_PROBES | 100 (random subsample) |

### Yields

- Indexed: 104,553 annotations across 24,352 images (sibling categories only)
- Raw distractor: 214 probes (before group cap)

| File | Probes | Pairs |
|---|---|---|
| `finegrained_distractor.json` | 160 | 80 |
| `finegrained_control.json` | 100 | — |

### Tier Breakdown

| Tier | Probes |
|---|---|
| `finegrained_confusable` | 104 |
| `finegrained_distinct` | 56 |

### Per-Group Distractor Counts (after cap)

| Group | Tier | Probes | % of total |
|---|---|---|---|
| bags | confusable | 40 | 25.0% |
| bowl_plate | distinct | 36 | 22.5% |
| cup_mug_glass | confusable | 34 | 21.2% |
| bottles | confusable | 12 | 7.5% |
| knife_scissors | distinct | 10 | 6.2% |
| utensils | confusable | 10 | 6.2% |
| headwear | distinct | 6 | 3.8% |
| cooking_vessels | confusable | 6 | 3.8% |
| seating | distinct | 4 | 2.5% |
| pillow_cushion | confusable | 2 | 1.2% |

### Notes

- LVIS categories are rare; co-occurrence is inherently sparse.
- Bags capped from 94 → 40 via pair-aware subsampling (seed=42).
- Phenomenon field encodes tier: `finegrained_confusable` vs `finegrained_distinct`.
- Report separately: "X on confusable siblings vs Y on distinct co-occurring categories".
- Prompt names use natural-language singular forms from LVIS synonyms.
- Mirrors linked via `pair_id` (e.g. `fg_0001`).
- Controls: one instance of a sibling category with NO other sibling in the image.

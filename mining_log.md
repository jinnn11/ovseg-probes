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

---

## Attribute-Binding Miner

**Date:** 2026-08-05
**Script:** `src/mine_attributes.py`
**Source:** `data/vg/attributes.json` + `data/vg/image_data.json`

### Design Decisions

- **Color normalization:** Only exact single-word matches against the fixed color
  set. Multi-word attributes ("dark blue", "light brown") are DROPPED, not
  normalized to base color. Avoids ambiguity at the cost of a few missed probes.
- **Object-name normalization:** Hand-built alias table merging VG synonyms
  (e.g., cup/mug → cup, car/automobile → car, pants/trousers/jeans → pants).
- **Conflict rule:** Two same-name objects with colors X ≠ Y, and neither
  object's attribute list contains the other's color.
- **No masks:** VG has no segmentation masks. `target_mask` is null for all
  attribute probes. Evaluated on boxes only.
- **Mirrors:** Each pair emits "the {X} {obj}" targeting object 1 and
  "the {Y} {obj}" targeting object 2, sharing a `pair_id`.

### Fixed Color Set

red, blue, green, yellow, black, white, brown, orange, pink, purple

### Thresholds

| Parameter | Value |
|---|---|
| MIN_BOX_AREA_FRAC | 0.005 (each box > 0.5% image area) |
| MAX_IOU | 0.10 (boxes must not overlap) |
| MAX_GROUP_PROBES | 30 (cap per object category) |
| MAX_DISTRACTOR_PAIRS | 200 (~400 probes total) |
| MAX_CONTROL_PROBES | 100 |

### Yields

- Raw distractor: 67,736 probes (massive VG coverage)
- After per-category cap + overall subsample:

| File | Probes | Pairs | Images |
|---|---|---|---|
| `attribute_distractor.json` | 400 | 200 | 193 |
| `attribute_control.json` | 100 | — | — |
| `attribute_extras.json` | 560 | 280 | — |

### Color Distribution (distractor)

| Color | Probes |
|---|---|
| blue | 76 |
| white | 68 |
| red | 65 |
| black | 56 |
| yellow | 37 |
| green | 36 |
| brown | 27 |
| orange | 14 |
| purple | 12 |
| pink | 9 |

### Notes

- 400 candidates expecting ~50% rejection at verification → ~200 keepers.
- Every kept probe gets human-verified in Step 7 (VG colors are noisy).
- Per-object cap at 30 probes ensures no single object category dominates.
- Prompt template: "the {color} {object}".
- Extras saved from post-cap, pre-subsample pool for topping up after verification.
- **Pair rejection policy:** keep valid singletons for accuracy; pair-consistency
  analysis uses only fully-intact pairs. Encoded in `src/analyze.py`.

---

## Negation Miner

**Date:** 2026-08-07
**Script:** `src/mine_negation.py`
**Sources:** `data/vg/relationships.json` + `data/vg/objects.json` + `data/vg/image_data.json`

### Design Decisions

- **Predicate normalization:** Variants of wearing/wears/wearing a/etc. mapped
  to canonical verbs (wearing, holding, carrying). Only verb+item combos that
  make sense are kept (e.g. "wearing a hat" yes, "wearing a phone" no).
- **Person detection:** Objects named person/man/woman/boy/girl/lady/guy.
- **Item normalization:** hat/cap → hat, glasses/sunglasses/eyeglasses → glasses,
  helmet kept separate, phone/cell phone → phone.
- **Probe structure:** Negation probe ("the person not wearing a hat") targets the
  unannotated person; positive mirror ("the person wearing a hat") targets the
  wearer. Shared pair_id gives negation-vs-affirmation contrast.
- **No masks:** VG has no segmentation masks. `target_mask` is null.
- **All probes flagged `needs_full_verification`:** Absence of VG annotation is
  weak evidence of absence. Heavy rejection expected at human verification.

### Target Items and Verbs

| Item | Verb |
|---|---|
| hat | wearing |
| helmet | wearing |
| glasses | wearing |
| backpack | wearing |
| tie | wearing |
| umbrella | holding |
| phone | holding |

### Thresholds

| Parameter | Value |
|---|---|
| MIN_BOX_AREA_FRAC | 0.015 (person box > 1.5% image area) |
| MAX_PERSON_IOU | 0.30 (overlapping people excluded) |
| MAX_DISTRACTOR_PAIRS | 150 (~300 probes total) |
| MAX_CONTROL_PROBES | 100 |
| PREFER_TWO_PERSON | True (2-person images prioritized) |

### Yields

- Raw distractor: 9,374 probes from VG relationship annotations
- Raw control: 154,389 probes (single-person, no item relation)
- After subsample:

| File | Probes | Pairs |
|---|---|---|
| `negation_distractor.json` | 300 (150 negation + 150 positive) | 150 |
| `negation_control.json` | 100 | — |
| `negation_extras.json` | 500 | 250 |

### Per-Item Distractor Counts

| Item | Probes |
|---|---|
| glasses | 120 |
| hat | 92 |
| helmet | 38 |
| umbrella | 20 |
| tie | 16 |
| backpack | 8 |
| phone | 6 |

### Accounting

- **150 negation-phrased probes** (phenomenon=`negation`) are the headline.
- **150 positive mirrors** (phenomenon=`negation_positive`) are a comparison
  condition, not padding for the negation count.
- Headline negation accuracy is computed from the 150 negation probes only;
  positives are reported separately for the negation-vs-affirmation contrast.
- Expect 150 pairs = 300 probes to verify (pair is verified together on same image).
- After rejection, negation phenomenon should land at ~100–130 keepers,
  topped up from 500 extras if needed.

### Notes

- All 300 distractor probes come from 2-person images (preferred for easier
  verification).
- Glasses and hat dominate due to VG annotation frequency; all 7 items present.
- Controls: single-person images, no item relationship annotated, same prompt.
- Extras available for top-up after verification rejects.
- Item distribution reflects VG bias — not rebalanced, since real-world
  frequency matters more than uniform coverage for this phenomenon.

---

## Step 7: Image Download + Verification

**Date:** 2026-08-07

### Scripts

| Script | Purpose |
|---|---|
| `src/download_images.py` | Concurrent image downloader with retries + PIL verification |
| `verify.ipynb` | Interactive verification notebook (ipywidgets, resume from decisions.jsonl) |
| `src/freeze.py` | Compile verified keepers into `probe_set_v1.json` + `control_set_v1.json` |

### Image Counts

| Source | Images |
|---|---|
| COCO/LVIS | 551 |
| Visual Genome | 1,007 |
| **Total** | **1,558** |

### Verification Session Plan

1. **Negation** — full review (300 distractor + 100 control, ~1.5 hr)
2. **Attributes** — full review (400 distractor + spot-check controls, ~1.5–2 hr)
3. **Spatial** — light pass (380 + 100, ~45 min)
4. **Fine-grained** — spot check 20% (~50 probes, ~20 min)
5. If spot-check exceeds ~5% bad → escalate to full review
6. Teammate re-verifies random 10% sample; compute agreement

### Freeze Process

After verification: `python -m src.freeze` compiles keepers, prints
per-phenomenon counts and pair integrity. Commit, tag `probe-freeze`.

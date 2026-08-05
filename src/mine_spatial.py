"""Mine spatial-relation probes from COCO train 2017 annotations.

Distractor set:
  Find images where category A appears exactly twice on opposite horizontal
  sides of a single instance of category B.  Emit two probes per image:
  "the {A} to the left of the {B}" and the mirror.

Control set:
  Same prompt structure, but A appears exactly once (no distractor).

All boxes are stored as xyxy pixels.  COCO xywh boxes are converted once
at load time via geometry.xywh_to_xyxy.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from pycocotools import mask as mask_utils

from src.geometry import box_area, box_center, xywh_to_xyxy
from src.schema import Probe, save_probes

# ── Allowed categories ──────────────────────────────────────────────
# ~20 common, unambiguous, mid-size COCO object categories.
# Excluded: person (too dominant), stuff-like, tiny objects.
ALLOWED_CATEGORIES: set[str] = {
    "bicycle", "car", "motorcycle", "bus", "truck",
    "dog", "cat", "horse", "cow", "sheep", "elephant", "bear",
    "backpack", "umbrella", "handbag", "suitcase",
    "bottle", "cup", "bowl",
    "chair", "couch", "potted plant",
    "laptop", "tv", "clock", "vase", "teddy bear",
}

# ── Geometric thresholds ────────────────────────────────────────────
MIN_HORIZ_GAP_FRAC = 0.30   # A centers horiz gap > 30 % of image width
MAX_VERT_DIFF_FRAC = 0.10   # A centers vert diff < 10 % of image height
MIN_BOX_AREA_FRAC = 0.03    # every box > 3 % of image area
MAX_BOX_AREA_FRAC = 0.25    # every box < 25 % of image area
MAX_CONTROL_PROBES = 100     # subsample controls (50 left + 50 right)

COCO_PATH = Path("data/coco/instances_train2017.json")
OUTPUT_DIR = Path("probes")


# ── Helpers ──────────────────────────────────────────────────────────

def _ann_to_rle(ann: dict, img_h: int, img_w: int) -> dict:
    """Convert a COCO annotation segmentation to compressed RLE."""
    seg = ann["segmentation"]
    if isinstance(seg, dict):
        # Already RLE (crowd annotations)
        return mask_utils.frPyObjects(seg, img_h, img_w)
    # Polygon list → merge into one RLE
    rles = mask_utils.frPyObjects(seg, img_h, img_w)
    return mask_utils.merge(rles)


def _rle_serializable(rle: dict) -> dict:
    """Make pycocotools RLE JSON-friendly (bytes counts → str)."""
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("utf-8")
    return {"counts": counts, "size": list(rle["size"])}


def _area_ok(box_xyxy: tuple, img_area: float) -> bool:
    frac = box_area(box_xyxy) / img_area
    return MIN_BOX_AREA_FRAC <= frac <= MAX_BOX_AREA_FRAC


# ── Core mining ─────────────────────────────────────────────────────

def mine(coco_path: Path = COCO_PATH) -> tuple[list[Probe], list[Probe]]:
    """Return (distractor_probes, control_probes)."""
    print(f"Loading {coco_path} …")
    with coco_path.open() as f:
        coco = json.load(f)

    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    img_id_to_info = {img["id"]: img for img in coco["images"]}

    # Index annotations by image (convert xywh → xyxy here, once)
    img_anns: dict[int, list[dict]] = defaultdict(list)
    for ann in coco["annotations"]:
        cat_name = cat_id_to_name.get(ann["category_id"], "")
        if cat_name not in ALLOWED_CATEGORIES:
            continue
        if ann.get("iscrowd", 0):
            continue
        img_anns[ann["image_id"]].append({
            "cat": cat_name,
            "box": xywh_to_xyxy(ann["bbox"]),
            "ann": ann,
        })

    print(f"Indexed {sum(len(v) for v in img_anns.values()):,} annotations "
          f"across {len(img_anns):,} images")

    distractor_probes: list[Probe] = []
    control_probes: list[Probe] = []
    pid = 0

    for image_id in sorted(img_anns):
        entries = img_anns[image_id]
        info = img_id_to_info[image_id]
        img_w, img_h = info["width"], info["height"]
        img_area = float(img_w * img_h)

        by_cat: dict[str, list[dict]] = defaultdict(list)
        for e in entries:
            by_cat[e["cat"]].append(e)

        # ── Distractor: A appears 2×, B appears 1× ──────────────
        for cat_a, a_list in by_cat.items():
            if len(a_list) != 2:
                continue
            for cat_b, b_list in by_cat.items():
                if cat_a == cat_b or len(b_list) != 1:
                    continue

                a1, a2 = a_list
                b = b_list[0]

                # Area filter
                if not all(_area_ok(e["box"], img_area) for e in [a1, a2, b]):
                    continue

                a1_cx, a1_cy = box_center(a1["box"])
                a2_cx, a2_cy = box_center(a2["box"])
                b_cx, _ = box_center(b["box"])

                # Both A's on opposite sides of B's center
                if not ((a1_cx < b_cx < a2_cx) or (a2_cx < b_cx < a1_cx)):
                    continue

                # Horizontal gap wide enough
                if abs(a1_cx - a2_cx) < MIN_HORIZ_GAP_FRAC * img_w:
                    continue

                # Vertical difference small enough (left/right, not above/below)
                if abs(a1_cy - a2_cy) > MAX_VERT_DIFF_FRAC * img_h:
                    continue

                # Sort into left / right
                if a1_cx < a2_cx:
                    left, right = a1, a2
                else:
                    left, right = a2, a1

                left_rle = _rle_serializable(
                    _ann_to_rle(left["ann"], img_h, img_w))
                right_rle = _rle_serializable(
                    _ann_to_rle(right["ann"], img_h, img_w))

                # Left probe
                pid += 1
                distractor_probes.append(Probe(
                    probe_id=f"spatial_{pid:04d}",
                    image_id=image_id,
                    image_source="coco_train2017",
                    phenomenon="spatial_left",
                    prompt=f"the {cat_a} to the left of the {cat_b}",
                    target_box=left["box"],
                    target_mask=left_rle,
                    distractor_box=right["box"],
                    has_distractor=True,
                    notes=f"cat_a={cat_a}, cat_b={cat_b}",
                ))

                # Right probe (mirror)
                pid += 1
                distractor_probes.append(Probe(
                    probe_id=f"spatial_{pid:04d}",
                    image_id=image_id,
                    image_source="coco_train2017",
                    phenomenon="spatial_right",
                    prompt=f"the {cat_a} to the right of the {cat_b}",
                    target_box=right["box"],
                    target_mask=right_rle,
                    distractor_box=left["box"],
                    has_distractor=True,
                    notes=f"cat_a={cat_a}, cat_b={cat_b}",
                ))

        # ── Control: A appears 1×, B appears 1× ─────────────────
        for cat_a, a_list in by_cat.items():
            if len(a_list) != 1:
                continue
            for cat_b, b_list in by_cat.items():
                if cat_a == cat_b or len(b_list) != 1:
                    continue

                a = a_list[0]
                b = b_list[0]

                if not all(_area_ok(e["box"], img_area) for e in [a, b]):
                    continue

                a_cx, a_cy = box_center(a["box"])
                b_cx, b_cy = box_center(b["box"])

                # Must be clearly left or right, not stacked
                if abs(a_cx - b_cx) < MIN_HORIZ_GAP_FRAC * img_w:
                    continue
                if abs(a_cy - b_cy) > MAX_VERT_DIFF_FRAC * img_h:
                    continue

                side = "left" if a_cx < b_cx else "right"

                a_rle = _rle_serializable(
                    _ann_to_rle(a["ann"], img_h, img_w))

                pid += 1
                control_probes.append(Probe(
                    probe_id=f"spatial_{pid:04d}",
                    image_id=image_id,
                    image_source="coco_train2017",
                    phenomenon=f"spatial_{side}_control",
                    prompt=f"the {cat_a} to the {side} of the {cat_b}",
                    target_box=a["box"],
                    target_mask=a_rle,
                    distractor_box=None,
                    has_distractor=False,
                    notes=f"cat_a={cat_a}, cat_b={cat_b}, control",
                ))

    # Subsample controls: stratified 50 left + 50 right
    if len(control_probes) > MAX_CONTROL_PROBES:
        left_ctrl = [p for p in control_probes if "left" in p.phenomenon]
        right_ctrl = [p for p in control_probes if "right" in p.phenomenon]
        half = MAX_CONTROL_PROBES // 2
        rng = random.Random(42)
        rng.shuffle(left_ctrl)
        rng.shuffle(right_ctrl)
        control_probes = left_ctrl[:half] + right_ctrl[:half]

    return distractor_probes, control_probes


# ── CLI ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mine spatial-relation probes from COCO train 2017")
    parser.add_argument("--coco", type=Path, default=COCO_PATH,
                        help="Path to instances_train2017.json")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help="Directory for output probe JSONs")
    args = parser.parse_args()

    distractor, control = mine(args.coco)

    print(f"\nDistractor probes: {len(distractor)}")
    print(f"Control probes:    {len(control)}")

    if distractor:
        p = args.output_dir / "spatial_distractor.json"
        save_probes(distractor, p)
        print(f"Saved → {p}")
    if control:
        p = args.output_dir / "spatial_control.json"
        save_probes(control, p)
        print(f"Saved → {p}")

    print(f"Total: {len(distractor) + len(control)} probes")


if __name__ == "__main__":
    main()

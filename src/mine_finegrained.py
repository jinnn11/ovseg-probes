"""Mine fine-grained category probes from LVIS v1 train annotations.

Distractor set:
  Find images containing exactly one instance each of two categories from
  the same confusable sibling group.  Emit one probe per instance: prompt
  is the target category name, distractor is the sibling.

Control set:
  Images with exactly one instance of a category and NO sibling from its
  group present.

All boxes are stored as xyxy pixels.  LVIS xywh boxes are converted once
at load time via geometry.xywh_to_xyxy.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from pycocotools import mask as mask_utils

from src.geometry import box_area, xywh_to_xyxy
from src.schema import Probe, save_probes

# ── Sibling groups ──────────────────────────────────────────────────
# Each group: (tier, members) where tier is "confusable" (a coarse model
# would collapse these into one concept) or "distinct" (clearly different
# objects that co-occur — serves as an easy-tier baseline).
# Members: list of (lvis_cat_id, natural_prompt_name).
SIBLING_GROUPS: dict[str, tuple[str, list[tuple[int, str]]]] = {
    "bags": ("confusable", [
        (35,  "handbag"),
        (34,  "backpack"),
        (36,  "suitcase"),
        (218, "tote bag"),
        (953, "shoulder bag"),
    ]),
    "cup_mug_glass": ("confusable", [
        (344, "cup"),
        (708, "mug"),
        (498, "drinking glass"),
        (1190, "wine glass"),
    ]),
    "bottles": ("confusable", [
        (1188, "wine bottle"),
        (1162, "water bottle"),
        (83,   "beer bottle"),
    ]),
    "utensils": ("confusable", [
        (993,  "spatula"),
        (1000, "spoon"),
        (1100, "tongs"),
        (622,  "ladle"),
        (1194, "wooden spoon"),
    ]),
    "cooking_vessels": ("confusable", [
        (836, "pot"),
        (477, "frying pan"),
        (751, "pan"),
        (1192, "wok"),
        (604,  "kettle"),
    ]),
    "pillow_cushion": ("confusable", [
        (804, "pillow"),
        (351, "cushion"),
    ]),
    "bowl_plate": ("distinct", [
        (139, "bowl"),
        (818, "plate"),
        (915, "saucer"),
        (819, "platter"),
    ]),
    "knife_scissors": ("distinct", [
        (615, "knife"),
        (923, "scissors"),
    ]),
    "headwear": ("distinct", [
        (544, "hat"),
        (556, "helmet"),
        (75,  "beanie"),
        (203, "cap"),
    ]),
    "seating": ("distinct", [
        (19,   "armchair"),
        (1018, "stool"),
        (232,  "chair"),
    ]),
}

MAX_GROUP_FRAC = 0.25  # cap any single group at 25 % of distractor total

# ── Thresholds ──────────────────────────────────────────────────────
MIN_BOX_AREA_FRAC = 0.01   # each box > 1 % of image area
MAX_BOX_AREA_FRAC = 0.50   # each box < 50 % of image area
MAX_CONTROL_PROBES = 100   # subsample controls

LVIS_PATH = Path("data/lvis/lvis_v1_train.json")
OUTPUT_DIR = Path("probes")


# ── Helpers ─────────────────────────────────────────────────────────

def _ann_to_rle(ann: dict, img_h: int, img_w: int) -> dict:
    seg = ann["segmentation"]
    if isinstance(seg, dict):
        return mask_utils.frPyObjects(seg, img_h, img_w)
    rles = mask_utils.frPyObjects(seg, img_h, img_w)
    return mask_utils.merge(rles)


def _rle_serializable(rle: dict) -> dict:
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("utf-8")
    return {"counts": counts, "size": list(rle["size"])}


def _area_ok(box_xyxy: tuple, img_area: float) -> bool:
    frac = box_area(box_xyxy) / img_area
    return MIN_BOX_AREA_FRAC <= frac <= MAX_BOX_AREA_FRAC


# ── Build lookups ───────────────────────────────────────────────────

def _build_cat_to_group() -> dict[int, str]:
    """Map each LVIS cat_id → its group name."""
    cat_to_group: dict[int, str] = {}
    for group_name, (_, members) in SIBLING_GROUPS.items():
        for cat_id, _ in members:
            cat_to_group[cat_id] = group_name
    return cat_to_group


def _build_cat_to_prompt() -> dict[int, str]:
    """Map each LVIS cat_id → its natural-language prompt name."""
    cat_to_prompt: dict[int, str] = {}
    for _, members in SIBLING_GROUPS.values():
        for cat_id, prompt_name in members:
            cat_to_prompt[cat_id] = prompt_name
    return cat_to_prompt


def _group_tier(group_name: str) -> str:
    """Return 'confusable' or 'distinct' for a group."""
    return SIBLING_GROUPS[group_name][0]


def _group_siblings(group_name: str) -> set[int]:
    """Return all cat_ids in a group."""
    return {cat_id for cat_id, _ in SIBLING_GROUPS[group_name][1]}


# ── Core mining ─────────────────────────────────────────────────────

def mine(
    lvis_path: Path = LVIS_PATH,
) -> tuple[list[Probe], list[Probe]]:
    """Return (distractor_probes, control_probes)."""
    print(f"Loading {lvis_path} …")
    with lvis_path.open() as f:
        lvis = json.load(f)

    img_id_to_info = {img["id"]: img for img in lvis["images"]}
    cat_to_group = _build_cat_to_group()
    cat_to_prompt = _build_cat_to_prompt()
    relevant_cat_ids = set(cat_to_group.keys())

    # Index: image_id → list of {cat_id, box_xyxy, ann}
    img_anns: dict[int, list[dict]] = defaultdict(list)
    for ann in lvis["annotations"]:
        cid = ann["category_id"]
        if cid not in relevant_cat_ids:
            continue
        img_anns[ann["image_id"]].append({
            "cat_id": cid,
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

        # Group entries by cat_id
        by_cat: dict[int, list[dict]] = defaultdict(list)
        for e in entries:
            by_cat[e["cat_id"]].append(e)

        # Which groups are represented in this image?
        groups_in_image: dict[str, list[int]] = defaultdict(list)
        for cid in by_cat:
            groups_in_image[cat_to_group[cid]].append(cid)

        for group_name, present_cats in groups_in_image.items():
            siblings = _group_siblings(group_name)

            # ── Distractor: exactly one each of two siblings ────
            for cat_a, cat_b in combinations(present_cats, 2):
                if len(by_cat[cat_a]) != 1 or len(by_cat[cat_b]) != 1:
                    continue

                ea = by_cat[cat_a][0]
                eb = by_cat[cat_b][0]

                if not (_area_ok(ea["box"], img_area) and
                        _area_ok(eb["box"], img_area)):
                    continue

                tier = _group_tier(group_name)
                phenom = f"finegrained_{tier}"

                # Probe for A (distractor = B)
                pid += 1
                pair_id = f"fg_{pid:04d}"
                a_rle = _rle_serializable(
                    _ann_to_rle(ea["ann"], img_h, img_w))
                distractor_probes.append(Probe(
                    probe_id=f"finegrained_{pid:04d}",
                    image_id=image_id,
                    image_source="lvis_v1_train",
                    phenomenon=phenom,
                    prompt=cat_to_prompt[cat_a],
                    target_box=ea["box"],
                    target_mask=a_rle,
                    distractor_box=eb["box"],
                    has_distractor=True,
                    pair_id=pair_id,
                    notes=f"group={group_name}, tier={tier}, "
                          f"target={cat_to_prompt[cat_a]}, "
                          f"distractor={cat_to_prompt[cat_b]}",
                ))

                # Probe for B (distractor = A)
                pid += 1
                b_rle = _rle_serializable(
                    _ann_to_rle(eb["ann"], img_h, img_w))
                distractor_probes.append(Probe(
                    probe_id=f"finegrained_{pid:04d}",
                    image_id=image_id,
                    image_source="lvis_v1_train",
                    phenomenon=phenom,
                    prompt=cat_to_prompt[cat_b],
                    target_box=eb["box"],
                    target_mask=b_rle,
                    distractor_box=ea["box"],
                    has_distractor=True,
                    pair_id=pair_id,
                    notes=f"group={group_name}, tier={tier}, "
                          f"target={cat_to_prompt[cat_b]}, "
                          f"distractor={cat_to_prompt[cat_a]}",
                ))

            # ── Control: exactly one instance, no sibling in image ──
            if len(present_cats) == 1:
                cat_c = present_cats[0]
                if len(by_cat[cat_c]) != 1:
                    continue
                ec = by_cat[cat_c][0]
                if not _area_ok(ec["box"], img_area):
                    continue

                c_rle = _rle_serializable(
                    _ann_to_rle(ec["ann"], img_h, img_w))
                pid += 1
                control_probes.append(Probe(
                    probe_id=f"finegrained_{pid:04d}",
                    image_id=image_id,
                    image_source="lvis_v1_train",
                    phenomenon="finegrained_control",
                    prompt=cat_to_prompt[cat_c],
                    target_box=ec["box"],
                    target_mask=c_rle,
                    distractor_box=None,
                    has_distractor=False,
                    notes=f"group={group_name}, target={cat_to_prompt[cat_c]}, control",
                ))

    # Cap any single group at MAX_GROUP_FRAC of total (pair-aware)
    distractor_probes = _cap_groups(distractor_probes)

    # Subsample controls
    if len(control_probes) > MAX_CONTROL_PROBES:
        rng = random.Random(42)
        rng.shuffle(control_probes)
        control_probes = control_probes[:MAX_CONTROL_PROBES]

    return distractor_probes, control_probes


def _cap_groups(probes: list[Probe]) -> list[Probe]:
    """Cap any single group so it doesn't exceed MAX_GROUP_FRAC of total.

    Iterates until no group exceeds the cap (capping one group lowers the
    total, which can push another group over the threshold).
    """
    by_group: dict[str, list[Probe]] = defaultdict(list)
    for p in probes:
        g = p.notes.split("group=")[1].split(",")[0]
        by_group[g].append(p)

    rng = random.Random(42)

    # Iterate until stable
    changed = True
    while changed:
        changed = False
        total = sum(len(gp) for gp in by_group.values())
        max_per_group = int(total * MAX_GROUP_FRAC)
        for group_name in list(by_group):
            gp = by_group[group_name]
            if len(gp) <= max_per_group:
                continue
            # Subsample by pair_id to keep mirrors together
            pair_ids = sorted(set(p.pair_id for p in gp))
            rng.shuffle(pair_ids)
            keep_pairs: set[str] = set()
            count = 0
            for pid in pair_ids:
                pair_size = sum(1 for p in gp if p.pair_id == pid)
                if count + pair_size > max_per_group:
                    break
                keep_pairs.add(pid)
                count += pair_size
            by_group[group_name] = [p for p in gp if p.pair_id in keep_pairs]
            changed = True

    kept: list[Probe] = []
    for gp in by_group.values():
        kept.extend(gp)
    return kept


# ── CLI ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mine fine-grained category probes from LVIS v1 train")
    parser.add_argument("--lvis", type=Path, default=LVIS_PATH,
                        help="Path to lvis_v1_train.json")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help="Directory for output probe JSONs")
    args = parser.parse_args()

    distractor, control = mine(args.lvis)

    print(f"\nDistractor probes: {len(distractor)}")
    print(f"Control probes:    {len(control)}")

    out = args.output_dir
    if distractor:
        save_probes(distractor, out / "finegrained_distractor.json")
        print(f"Saved → {out / 'finegrained_distractor.json'}")
    if control:
        save_probes(control, out / "finegrained_control.json")
        print(f"Saved → {out / 'finegrained_control.json'}")

    # Per-group breakdown
    from collections import Counter
    group_counts = Counter()
    for p in distractor:
        g = p.notes.split("group=")[1].split(",")[0]
        group_counts[g] += 1
    print("\nPer-group distractor counts:")
    for g, cnt in group_counts.most_common():
        print(f"  {g:<20} {cnt}")

    print(f"\nTotal: {len(distractor) + len(control)} probes")


if __name__ == "__main__":
    main()

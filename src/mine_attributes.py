"""Mine attribute-binding probes from Visual Genome attributes.json.

Distractor set:
  Find images with two objects of the same normalized name where object 1
  has color X, object 2 has color Y, X != Y, and neither object lists the
  other's color.  Emit mirrored probes sharing a pair_id.

Control set:
  Images with exactly one instance of a colored object category and no
  same-name object present.

All boxes are stored as xyxy pixels.  VG boxes (x, y, w, h) are converted
once at load time.  No masks (VG has none) — target_mask is null.

Color normalization: multi-word attributes like "dark blue" or "light brown"
are DROPPED (not normalized to base color) to avoid ambiguity.  Only exact
single-word matches against the fixed color set are used.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from src.geometry import box_area, box_iou, xywh_to_xyxy
from src.schema import Probe, save_probes

# ── Fixed color set ─────────────────────────────────────────────────
COLORS: set[str] = {
    "red", "blue", "green", "yellow", "black", "white",
    "brown", "orange", "pink", "purple",
}

# ── Object-name normalization ───────────────────────────────────────
# Merge common VG synonyms into a canonical name.
# Only include objects that commonly carry color attributes.
NAME_ALIASES: dict[str, str] = {
    "car": "car", "cars": "car", "automobile": "car", "vehicle": "car",
    "truck": "truck", "trucks": "truck",
    "bus": "bus", "buses": "bus",
    "shirt": "shirt", "t-shirt": "shirt", "tee shirt": "shirt",
    "jacket": "jacket", "coat": "jacket",
    "pants": "pants", "trousers": "pants", "jeans": "pants",
    "shorts": "shorts",
    "hat": "hat", "cap": "hat",
    "helmet": "helmet",
    "shoe": "shoe", "shoes": "shoe", "sneaker": "shoe", "sneakers": "shoe",
    "bag": "bag", "handbag": "bag", "purse": "bag", "backpack": "bag",
    "umbrella": "umbrella", "umbrellas": "umbrella",
    "chair": "chair", "chairs": "chair",
    "cup": "cup", "mug": "cup", "cups": "cup",
    "plate": "plate", "plates": "plate", "dish": "plate",
    "bowl": "bowl", "bowls": "bowl",
    "bottle": "bottle", "bottles": "bottle",
    "flower": "flower", "flowers": "flower",
    "door": "door", "doors": "door",
    "sign": "sign", "signs": "sign",
    "building": "building", "buildings": "building",
    "fence": "fence",
    "ball": "ball",
    "box": "box", "boxes": "box",
    "pillow": "pillow", "pillows": "pillow", "cushion": "pillow",
    "towel": "towel", "towels": "towel",
    "flag": "flag", "flags": "flag",
    "boat": "boat", "boats": "boat",
    "kite": "kite", "kites": "kite",
    "suitcase": "suitcase", "luggage": "suitcase",
    "surfboard": "surfboard",
    "vase": "vase",
}

# ── Thresholds ──────────────────────────────────────────────────────
MIN_BOX_AREA_FRAC = 0.005   # each box > 0.5 % of image area
MAX_IOU = 0.10               # boxes must not overlap much
MAX_DISTRACTOR_PAIRS = 200   # ~400 probes total (pair-aware subsample)
MAX_CONTROL_PROBES = 100
MAX_GROUP_PROBES = 30        # cap any single object category

VG_ATTRS_PATH = Path("data/vg/attributes.json")
VG_IMGDATA_PATH = Path("data/vg/image_data.json")
OUTPUT_DIR = Path("probes")


# ── Helpers ─────────────────────────────────────────────────────────

def _normalize_name(names: list[str]) -> str | None:
    """Return canonical object name, or None if not in our alias table."""
    for n in names:
        key = n.lower().strip()
        if key in NAME_ALIASES:
            return NAME_ALIASES[key]
    return None


def _extract_colors(attributes: list[str]) -> set[str]:
    """Return the set of exact-match colors from an object's attributes."""
    return {a.lower().strip() for a in attributes} & COLORS


def _vg_box_to_xyxy(obj: dict) -> tuple[float, float, float, float]:
    """Convert VG object dict (x, y, w, h) to xyxy."""
    return xywh_to_xyxy([obj["x"], obj["y"], obj["w"], obj["h"]])


def _area_ok(box_xyxy: tuple, img_area: float) -> bool:
    return box_area(box_xyxy) / img_area >= MIN_BOX_AREA_FRAC


# ── Core mining ─────────────────────────────────────────────────────

def mine(
    attrs_path: Path = VG_ATTRS_PATH,
    imgdata_path: Path = VG_IMGDATA_PATH,
) -> tuple[list[Probe], list[Probe]]:
    """Return (distractor_probes, control_probes)."""
    print(f"Loading {imgdata_path} …")
    with imgdata_path.open() as f:
        img_data_list = json.load(f)
    img_dims = {img["image_id"]: (img["width"], img["height"])
                for img in img_data_list}

    print(f"Loading {attrs_path} …")
    with attrs_path.open() as f:
        vg_attrs = json.load(f)
    print(f"Loaded {len(vg_attrs):,} images")

    distractor_probes: list[Probe] = []
    control_probes: list[Probe] = []
    pid = 0

    for img_entry in vg_attrs:
        image_id = img_entry["image_id"]
        dims = img_dims.get(image_id)
        if dims is None:
            continue
        img_w, img_h = dims
        img_area = float(img_w * img_h)
        if img_area == 0:
            continue

        # Parse objects: keep those with a normalized name and at least one color
        parsed: list[dict] = []
        for obj in img_entry.get("attributes", []):
            norm_name = _normalize_name(obj.get("names", []))
            if norm_name is None:
                continue
            colors = _extract_colors(obj.get("attributes", []))
            if not colors:
                continue
            try:
                box = _vg_box_to_xyxy(obj)
            except (ValueError, KeyError):
                continue
            if not _area_ok(box, img_area):
                continue
            parsed.append({
                "name": norm_name,
                "colors": colors,
                "box": box,
            })

        # Group by normalized name
        by_name: dict[str, list[dict]] = defaultdict(list)
        for p in parsed:
            by_name[p["name"]].append(p)

        # ── Distractor: two same-name objects, different exclusive colors
        for obj_name, instances in by_name.items():
            if len(instances) < 2:
                continue
            for a, b in combinations(instances, 2):
                # Find exclusive color pairs: a has X not in b, b has Y not in a
                a_exclusive = a["colors"] - b["colors"]
                b_exclusive = b["colors"] - a["colors"]
                if not a_exclusive or not b_exclusive:
                    continue

                # Check IoU
                try:
                    iou = box_iou(a["box"], b["box"])
                except ValueError:
                    continue
                if iou > MAX_IOU:
                    continue

                # Pick one color per object (deterministic: sorted first)
                color_a = sorted(a_exclusive)[0]
                color_b = sorted(b_exclusive)[0]

                pid += 1
                pair_id = f"attr_{pid:04d}"

                # Probe for A: "the {color_a} {obj_name}"
                distractor_probes.append(Probe(
                    probe_id=f"attribute_{pid:04d}",
                    image_id=image_id,
                    image_source="visual_genome",
                    phenomenon="attribute_color",
                    prompt=f"the {color_a} {obj_name}",
                    target_box=a["box"],
                    target_mask=None,
                    distractor_box=b["box"],
                    has_distractor=True,
                    pair_id=pair_id,
                    notes=f"color_a={color_a}, color_b={color_b}, "
                          f"object={obj_name}",
                ))

                # Probe for B: "the {color_b} {obj_name}"
                pid += 1
                distractor_probes.append(Probe(
                    probe_id=f"attribute_{pid:04d}",
                    image_id=image_id,
                    image_source="visual_genome",
                    phenomenon="attribute_color",
                    prompt=f"the {color_b} {obj_name}",
                    target_box=b["box"],
                    target_mask=None,
                    distractor_box=a["box"],
                    has_distractor=True,
                    pair_id=pair_id,
                    notes=f"color_a={color_b}, color_b={color_a}, "
                          f"object={obj_name}",
                ))

        # ── Control: exactly one instance of a named category, with color
        for obj_name, instances in by_name.items():
            if len(instances) != 1:
                continue
            obj = instances[0]
            color = sorted(obj["colors"])[0]
            pid += 1
            control_probes.append(Probe(
                probe_id=f"attribute_{pid:04d}",
                image_id=image_id,
                image_source="visual_genome",
                phenomenon="attribute_color_control",
                prompt=f"the {color} {obj_name}",
                target_box=obj["box"],
                target_mask=None,
                distractor_box=None,
                has_distractor=False,
                notes=f"color={color}, object={obj_name}, control",
            ))

    print(f"Raw distractor probes: {len(distractor_probes)}")
    print(f"Raw control probes:    {len(control_probes)}")

    # Cap per-object-category to avoid skew
    distractor_probes = _cap_by_object(distractor_probes)

    # Overall subsample to target size (pair-aware)
    all_pair_ids = sorted(set(p.pair_id for p in distractor_probes))
    if len(all_pair_ids) > MAX_DISTRACTOR_PAIRS:
        rng = random.Random(42)
        rng.shuffle(all_pair_ids)
        keep = set(all_pair_ids[:MAX_DISTRACTOR_PAIRS])
        distractor_probes = [p for p in distractor_probes if p.pair_id in keep]

    # Subsample controls
    if len(control_probes) > MAX_CONTROL_PROBES:
        rng = random.Random(42)
        rng.shuffle(control_probes)
        control_probes = control_probes[:MAX_CONTROL_PROBES]

    return distractor_probes, control_probes


def _cap_by_object(probes: list[Probe]) -> list[Probe]:
    """Cap any single object category at MAX_GROUP_PROBES (pair-aware)."""
    by_obj: dict[str, list[Probe]] = defaultdict(list)
    for p in probes:
        obj = p.notes.split("object=")[1].strip()
        by_obj[obj].append(p)

    rng = random.Random(42)
    kept: list[Probe] = []
    for obj_name, obj_probes in by_obj.items():
        if len(obj_probes) <= MAX_GROUP_PROBES:
            kept.extend(obj_probes)
            continue
        pair_ids = sorted(set(p.pair_id for p in obj_probes))
        rng.shuffle(pair_ids)
        keep_pairs: set[str] = set()
        count = 0
        for pid in pair_ids:
            pair_size = sum(1 for p in obj_probes if p.pair_id == pid)
            if count + pair_size > MAX_GROUP_PROBES:
                break
            keep_pairs.add(pid)
            count += pair_size
        kept.extend(p for p in obj_probes if p.pair_id in keep_pairs)
    return kept


# ── CLI ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mine attribute-binding probes from VG attributes")
    parser.add_argument("--attrs", type=Path, default=VG_ATTRS_PATH)
    parser.add_argument("--imgdata", type=Path, default=VG_IMGDATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    distractor, control = mine(args.attrs, args.imgdata)

    out = args.output_dir
    if distractor:
        save_probes(distractor, out / "attribute_distractor.json")
        print(f"Saved → {out / 'attribute_distractor.json'}")
    if control:
        save_probes(control, out / "attribute_control.json")
        print(f"Saved → {out / 'attribute_control.json'}")

    # Per-object breakdown
    from collections import Counter
    obj_counts = Counter()
    for p in distractor:
        obj = p.notes.split("object=")[1].strip()
        obj_counts[obj] += 1
    print(f"\nDistractor probes: {len(distractor)}")
    print(f"Control probes:    {len(control)}")
    print("\nPer-object distractor counts:")
    for obj, cnt in obj_counts.most_common(15):
        print(f"  {obj:<20} {cnt}")

    pairs = set(p.pair_id for p in distractor)
    print(f"\nTotal pairs: {len(pairs)}")
    print(f"Total: {len(distractor) + len(control)} probes")


if __name__ == "__main__":
    main()

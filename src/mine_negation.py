"""Mine negation probes from Visual Genome relationships.json + objects.json.

Distractor set:
  Images with 2–4 persons where at least one person wears/holds/carries
  a target item and at least one person has no such relationship with that
  item.  Emit "the person not wearing a {item}" targeting the unannotated
  person, with the wearer as distractor.  Also emit the positive mirror
  "the person wearing a {item}" targeting the wearer, shared pair_id.

Control set:
  Single-person images where the person has no relationship involving the
  target item.  Prompt unchanged — "the person not wearing a {item}".

All probes flagged needs_full_verification: absence of VG annotation is
weak evidence of absence.  Heavy rejection expected at verification.

Boxes are xyxy pixels.  VG uses xywh — converted at load time.  No masks.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from src.geometry import box_area, box_iou, xywh_to_xyxy
from src.schema import Probe, save_probes

# ── Predicate normalization ────────────────────────────────────────
# Map common VG predicate variants to canonical verb.
PRED_ALIASES: dict[str, str] = {}
for _canon, _variants in {
    "wearing": ["wearing", "wears", "wearing a", "wearing an",
                "wears a", "wear", "is wearing", "are wearing"],
    "holding": ["holding", "holds", "holding a", "holding an",
                "hold", "holds a", "is holding", "are holding"],
    "carrying": ["carrying", "carries", "carrying a", "carrying an",
                 "carry", "is carrying", "are carrying"],
}.items():
    for v in _variants:
        PRED_ALIASES[v] = _canon

# ── Person aliases ─────────────────────────────────────────────────
PERSON_NAMES: set[str] = {"person", "man", "woman", "boy", "girl", "lady", "guy"}

# ── Target items and normalization ─────────────────────────────────
ITEM_ALIASES: dict[str, str] = {
    "hat": "hat", "cap": "hat",
    "helmet": "helmet",
    "glasses": "glasses", "sunglasses": "glasses", "eyeglasses": "glasses",
    "backpack": "backpack",
    "umbrella": "umbrella",
    "tie": "tie", "necktie": "tie",
    "phone": "phone", "cell phone": "phone", "cellphone": "phone",
}
TARGET_ITEMS: set[str] = set(ITEM_ALIASES.values())

# Prompt verb per item (wearing vs holding vs carrying)
ITEM_VERB: dict[str, str] = {
    "hat": "wearing",
    "helmet": "wearing",
    "glasses": "wearing",
    "backpack": "wearing",
    "tie": "wearing",
    "umbrella": "holding",
    "phone": "holding",
}

# Article per item ("a hat" vs bare "glasses")
ITEM_ARTICLE: dict[str, str] = {
    "hat": "a ",
    "helmet": "a ",
    "glasses": "",
    "backpack": "a ",
    "umbrella": "an ",
    "tie": "a ",
    "phone": "a ",
}

# ── Thresholds ─────────────────────────────────────────────────────
MIN_BOX_AREA_FRAC = 0.015    # person box > 1.5% of image area
MAX_PERSON_IOU = 0.30         # overlapping persons are unjudgeable
MAX_DISTRACTOR_PAIRS = 150    # ~300 probes (mirrored)
MAX_EXTRAS = 500
MAX_CONTROL_PROBES = 100
PREFER_TWO_PERSON = True      # weight 2-person images higher

VG_RELS_PATH = Path("data/vg/relationships.json")
VG_OBJS_PATH = Path("data/vg/objects.json")
VG_IMGDATA_PATH = Path("data/vg/image_data.json")
OUTPUT_DIR = Path("probes")


# ── Helpers ────────────────────────────────────────────────────────

def _get_name(entity: dict) -> str:
    """Extract lowercase name from a VG subject/object dict."""
    if "name" in entity:
        n = entity["name"]
        if isinstance(n, str):
            return n.lower().strip()
        if isinstance(n, list) and n:
            return n[0].lower().strip()
    if "names" in entity:
        ns = entity["names"]
        if isinstance(ns, list) and ns:
            return ns[0].lower().strip()
    return ""


def _entity_box(entity: dict) -> tuple[float, float, float, float]:
    """Convert VG entity (x, y, w, h) to xyxy."""
    return xywh_to_xyxy([entity["x"], entity["y"], entity["w"], entity["h"]])


def _area_ok(box: tuple, img_area: float) -> bool:
    return box_area(box) / img_area >= MIN_BOX_AREA_FRAC


def _all_pairs_separated(boxes: list[tuple], max_iou: float) -> bool:
    """Check that all pairwise IoUs are below the threshold."""
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if box_iou(boxes[i], boxes[j]) > max_iou:
                return False
    return True


# ── Data loading ───────────────────────────────────────────────────

def _load_image_dims(path: Path) -> dict[int, tuple[int, int]]:
    print(f"Loading {path} …")
    with path.open() as f:
        data = json.load(f)
    return {img["image_id"]: (img["width"], img["height"]) for img in data}


def _load_person_objects(objs_path: Path) -> dict[int, list[dict]]:
    """Return {image_id: [{object_id, name, box}, ...]} for person objects."""
    print(f"Loading {objs_path} …")
    with objs_path.open() as f:
        data = json.load(f)
    result: dict[int, list[dict]] = {}
    for img_entry in data:
        image_id = img_entry["image_id"]
        persons = []
        for obj in img_entry.get("objects", []):
            names = obj.get("names", [])
            if not names:
                n = obj.get("name", "")
                names = [n] if n else []
            for n in names:
                if n.lower().strip() in PERSON_NAMES:
                    try:
                        box = _entity_box(obj)
                    except (ValueError, KeyError):
                        break
                    persons.append({
                        "object_id": obj["object_id"],
                        "name": n.lower().strip(),
                        "box": box,
                    })
                    break
        if persons:
            result[image_id] = persons
    return result


def _load_person_item_rels(
    rels_path: Path,
) -> dict[int, list[dict]]:
    """Return {image_id: [{subject_id, item, verb, subj_box, obj_box}, ...]}.

    Only relationships where a person-type subject wears/holds/carries a
    target item are kept.
    """
    print(f"Loading {rels_path} …")
    with rels_path.open() as f:
        data = json.load(f)

    result: dict[int, list[dict]] = defaultdict(list)
    for img_entry in data:
        image_id = img_entry["image_id"]
        for rel in img_entry.get("relationships", []):
            pred = rel["predicate"].lower().strip()
            canon_verb = PRED_ALIASES.get(pred)
            if canon_verb is None:
                continue

            subj_name = _get_name(rel["subject"])
            if subj_name not in PERSON_NAMES:
                continue

            obj_name = _get_name(rel["object"])
            canon_item = ITEM_ALIASES.get(obj_name)
            if canon_item is None:
                continue

            expected_verb = ITEM_VERB.get(canon_item, "wearing")
            if canon_verb != expected_verb:
                continue

            result[image_id].append({
                "subject_id": rel["subject"]["object_id"],
                "item": canon_item,
                "verb": canon_verb,
            })

    return dict(result)


# ── Core mining ────────────────────────────────────────────────────

def mine(
    rels_path: Path = VG_RELS_PATH,
    objs_path: Path = VG_OBJS_PATH,
    imgdata_path: Path = VG_IMGDATA_PATH,
) -> tuple[list[Probe], list[Probe], list[Probe]]:
    """Return (distractor_probes, control_probes, extras_probes)."""

    img_dims = _load_image_dims(imgdata_path)
    person_objs = _load_person_objects(objs_path)
    person_item_rels = _load_person_item_rels(rels_path)

    distractor_probes: list[Probe] = []
    control_probes: list[Probe] = []
    pid = 0

    for image_id, persons in person_objs.items():
        dims = img_dims.get(image_id)
        if dims is None:
            continue
        img_w, img_h = dims
        img_area = float(img_w * img_h)
        if img_area == 0:
            continue

        # Filter persons by box area
        valid_persons = [
            p for p in persons
            if _area_ok(p["box"], img_area)
        ]
        if not valid_persons:
            continue

        # Check pairwise IoU — skip image if any pair overlaps
        if len(valid_persons) > 1:
            boxes = [p["box"] for p in valid_persons]
            if not _all_pairs_separated(boxes, MAX_PERSON_IOU):
                continue

        # Get item relationships for this image
        rels = person_item_rels.get(image_id, [])
        # Build {person_object_id: set of items they have}
        person_items: dict[int, set[str]] = defaultdict(set)
        for r in rels:
            person_items[r["subject_id"]].add(r["item"])

        n_persons = len(valid_persons)

        # ── DISTRACTOR: 2–4 persons, ≥1 has item, ≥1 lacks it
        if 2 <= n_persons <= 4:
            for item in TARGET_ITEMS:
                verb = ITEM_VERB.get(item, "wearing")

                wearers = [p for p in valid_persons
                           if item in person_items.get(p["object_id"], set())]
                non_wearers = [p for p in valid_persons
                               if item not in person_items.get(p["object_id"], set())]

                if not wearers or not non_wearers:
                    continue

                # Prefer 2-person images: take exactly one of each
                if n_persons == 2:
                    target = non_wearers[0]
                    distractor_person = wearers[0]
                else:
                    # Multi-person: pick the most isolated pair
                    target = non_wearers[0]
                    distractor_person = wearers[0]

                pid += 1
                pair_id = f"neg_{pid:04d}"
                article = ITEM_ARTICLE.get(item, "a ")

                # Negation probe: "the person not wearing a hat"
                distractor_probes.append(Probe(
                    probe_id=f"negation_{pid:04d}",
                    image_id=image_id,
                    image_source="visual_genome",
                    phenomenon="negation",
                    prompt=f"the person not {verb} {article}{item}",
                    target_box=target["box"],
                    target_mask=None,
                    distractor_box=distractor_person["box"],
                    has_distractor=True,
                    pair_id=pair_id,
                    notes=f"item={item}, verb={verb}, "
                          f"n_persons={n_persons}, needs_full_verification",
                ))

                # Positive mirror: "the person wearing a hat"
                pid += 1
                distractor_probes.append(Probe(
                    probe_id=f"negation_{pid:04d}",
                    image_id=image_id,
                    image_source="visual_genome",
                    phenomenon="negation_positive",
                    prompt=f"the person {verb} {article}{item}",
                    target_box=distractor_person["box"],
                    target_mask=None,
                    distractor_box=target["box"],
                    has_distractor=True,
                    pair_id=pair_id,
                    notes=f"item={item}, verb={verb}, positive_mirror, "
                          f"n_persons={n_persons}, needs_full_verification",
                ))

        # ── CONTROL: single person, no relationship with any target item
        if n_persons == 1:
            person = valid_persons[0]
            items_on = person_items.get(person["object_id"], set())
            for item in TARGET_ITEMS:
                if item in items_on:
                    continue
                verb = ITEM_VERB.get(item, "wearing")
                article = ITEM_ARTICLE.get(item, "a ")
                pid += 1
                control_probes.append(Probe(
                    probe_id=f"negation_{pid:04d}",
                    image_id=image_id,
                    image_source="visual_genome",
                    phenomenon="negation_control",
                    prompt=f"the person not {verb} {article}{item}",
                    target_box=person["box"],
                    target_mask=None,
                    distractor_box=None,
                    has_distractor=False,
                    notes=f"item={item}, verb={verb}, control, "
                          f"needs_full_verification",
                ))

    print(f"Raw distractor probes: {len(distractor_probes)}")
    print(f"Raw control probes:    {len(control_probes)}")

    # Subsample distractor (pair-aware), preferring 2-person images
    all_pair_ids = sorted(set(p.pair_id for p in distractor_probes))
    extras_probes: list[Probe] = []

    if len(all_pair_ids) > MAX_DISTRACTOR_PAIRS:
        rng = random.Random(42)

        # Partition into 2-person and 3-4-person pairs
        pair_persons: dict[str, int] = {}
        for p in distractor_probes:
            if p.pair_id and p.pair_id not in pair_persons:
                for part in p.notes.split(","):
                    part = part.strip()
                    if part.startswith("n_persons="):
                        pair_persons[p.pair_id] = int(part.split("=")[1])
                        break

        two_person = [pid for pid in all_pair_ids
                      if pair_persons.get(pid, 0) == 2]
        multi_person = [pid for pid in all_pair_ids
                        if pair_persons.get(pid, 0) > 2]
        rng.shuffle(two_person)
        rng.shuffle(multi_person)

        # Take 2-person first, then fill with multi-person
        selected: list[str] = []
        selected.extend(two_person[:MAX_DISTRACTOR_PAIRS])
        remaining = MAX_DISTRACTOR_PAIRS - len(selected)
        if remaining > 0:
            selected.extend(multi_person[:remaining])

        keep = set(selected[:MAX_DISTRACTOR_PAIRS])
        leftover_ids = [pid for pid in all_pair_ids if pid not in keep]

        # Save extras
        extras_pool = [p for p in distractor_probes
                       if p.pair_id in set(leftover_ids)]
        if len(extras_pool) > MAX_EXTRAS:
            extra_pairs = sorted(set(p.pair_id for p in extras_pool))
            rng2 = random.Random(42)
            rng2.shuffle(extra_pairs)
            keep_extra: set[str] = set()
            count = 0
            for epid in extra_pairs:
                pair_size = sum(1 for p in extras_pool if p.pair_id == epid)
                if count + pair_size > MAX_EXTRAS:
                    break
                keep_extra.add(epid)
                count += pair_size
            extras_pool = [p for p in extras_pool if p.pair_id in keep_extra]
        extras_probes = extras_pool

        distractor_probes = [p for p in distractor_probes if p.pair_id in keep]

    # Subsample controls
    if len(control_probes) > MAX_CONTROL_PROBES:
        rng = random.Random(42)
        rng.shuffle(control_probes)
        control_probes = control_probes[:MAX_CONTROL_PROBES]

    return distractor_probes, control_probes, extras_probes


# ── CLI ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mine negation probes from VG relationships + objects")
    parser.add_argument("--rels", type=Path, default=VG_RELS_PATH)
    parser.add_argument("--objs", type=Path, default=VG_OBJS_PATH)
    parser.add_argument("--imgdata", type=Path, default=VG_IMGDATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    distractor, control, extras = mine(args.rels, args.objs, args.imgdata)

    out = args.output_dir
    if distractor:
        save_probes(distractor, out / "negation_distractor.json")
        print(f"Saved → {out / 'negation_distractor.json'}")
    if control:
        save_probes(control, out / "negation_control.json")
        print(f"Saved → {out / 'negation_control.json'}")
    if extras:
        save_probes(extras, out / "negation_extras.json")
        print(f"Saved → {out / 'negation_extras.json'}")

    # Summary
    from collections import Counter
    item_counts = Counter()
    n_person_counts = Counter()
    positive_count = 0
    for p in distractor:
        for part in p.notes.split(","):
            part = part.strip()
            if part.startswith("item="):
                item_counts[part.split("=")[1]] += 1
            if part.startswith("n_persons="):
                n_person_counts[int(part.split("=")[1])] += 1
        if "positive_mirror" in p.notes:
            positive_count += 1

    print(f"\nDistractor probes: {len(distractor)} "
          f"({positive_count} positive mirrors)")
    print(f"Control probes:    {len(control)}")

    print("\nPer-item distractor counts:")
    for item, cnt in item_counts.most_common():
        print(f"  {item:<15} {cnt}")

    print("\nBy number of persons in image:")
    for n, cnt in sorted(n_person_counts.items()):
        print(f"  {n}-person: {cnt}")

    pairs = set(p.pair_id for p in distractor)
    extra_pairs = set(p.pair_id for p in extras)
    print(f"\nTotal pairs: {len(pairs)}")
    print(f"Extras: {len(extras)} probes ({len(extra_pairs)} pairs)")
    print(f"Total: {len(distractor) + len(control)} probes "
          f"(+ {len(extras)} extras)")


if __name__ == "__main__":
    main()

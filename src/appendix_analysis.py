"""Per-group appendix analyses for the paper.

Produces:
  1. Per-color-pair attribute breakdown
  2. Per-item negation table
  3. Per-group fine-grained table
  4. Spatial other-grounding spot-check (what got grabbed?)

Usage:
    python -m src.appendix_analysis --predictions-dir predictions/grounded_sam
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from src.geometry import box_iou
from src.schema import Probe, load_probes


def _load_pred(pred_dir: Path, probe_id: str) -> dict | None:
    path = pred_dir / f"{probe_id}.json"
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def _is_correct(probe: Probe, pred: dict, thr: float = 0.5) -> bool:
    top_box = pred.get("top_box")
    if top_box is None:
        return False
    if box_iou(probe.target_box, top_box) < thr:
        return False
    if probe.distractor_box is not None:
        if box_iou(probe.distractor_box, top_box) >= thr:
            return False
    return True


def _classify_failure(probe: Probe, pred: dict, thr: float = 0.5) -> str:
    top_box = pred.get("top_box")
    if top_box is None:
        return "detection-miss"
    if probe.distractor_box is not None:
        if box_iou(probe.distractor_box, top_box) >= thr:
            return "distractor-capture"
    if box_iou(probe.target_box, top_box) < thr:
        return "other-grounding"
    return "correct"


def _extract_color(prompt: str) -> str | None:
    colors = ["red", "blue", "green", "yellow", "black", "white",
              "brown", "orange", "pink", "purple"]
    for c in colors:
        if f"the {c} " in prompt:
            return c
    return None


def _extract_negation_item(prompt: str) -> str | None:
    m = re.search(r"(?:not )?(?:wearing|holding|carrying) (?:a |an )?(\w+)", prompt)
    if m:
        return m.group(1)
    return None


def _extract_fg_group(probe: Probe) -> str:
    return probe.notes if probe.notes else "unknown"


def attribute_breakdown(probes: list[Probe], pred_dir: Path, out_dir: Path) -> None:
    print("\n=== ATTRIBUTE COLOR BREAKDOWN ===")
    color_stats: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0,
                                                         "captures": 0, "other": 0})
    pair_colors: dict[str, list[str]] = defaultdict(list)

    attr_probes = [p for p in probes if p.phenomenon == "attribute_color"]

    for probe in attr_probes:
        pred = _load_pred(pred_dir, probe.probe_id)
        if pred is None:
            continue
        color = _extract_color(probe.prompt)
        if color is None:
            continue

        stats = color_stats[color]
        stats["total"] += 1
        failure = _classify_failure(probe, pred)
        if failure == "correct":
            stats["correct"] += 1
        elif failure == "distractor-capture":
            stats["captures"] += 1
        elif failure == "other-grounding":
            stats["other"] += 1

        if probe.pair_id:
            pair_colors[probe.pair_id].append(color)

    rows = []
    for color in sorted(color_stats.keys()):
        s = color_stats[color]
        acc = s["correct"] / s["total"] * 100 if s["total"] > 0 else 0
        rows.append((color, s["correct"], s["total"], acc, s["captures"], s["other"]))
        print(f"  {color:8s}  {s['correct']:3d}/{s['total']:3d} = {acc:5.1f}%  "
              f"captures={s['captures']}  other={s['other']}")

    out_path = out_dir / "attribute_per_color.csv"
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["color", "correct", "total", "accuracy", "distractor_capture", "other_grounding"])
        w.writerows(rows)

    # color-pair analysis
    print("\n  Color-pair accuracy (target_color vs distractor_color):")
    pair_acc: dict[tuple, dict] = defaultdict(lambda: {"correct": 0, "total": 0})
    for pair_id, colors in pair_colors.items():
        if len(colors) != 2:
            continue
        c1, c2 = sorted(colors)
        pair_key = (c1, c2)
        for probe in attr_probes:
            if probe.pair_id != pair_id:
                continue
            pred = _load_pred(pred_dir, probe.probe_id)
            if pred is None:
                continue
            pair_acc[pair_key]["total"] += 1
            if _is_correct(probe, pred):
                pair_acc[pair_key]["correct"] += 1

    pair_rows = []
    for (c1, c2) in sorted(pair_acc.keys()):
        s = pair_acc[(c1, c2)]
        acc = s["correct"] / s["total"] * 100 if s["total"] > 0 else 0
        pair_rows.append((c1, c2, s["correct"], s["total"], acc))
        print(f"    {c1:8s} vs {c2:8s}  {s['correct']:3d}/{s['total']:3d} = {acc:5.1f}%")

    out_path = out_dir / "attribute_per_color_pair.csv"
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["color_1", "color_2", "correct", "total", "accuracy"])
        w.writerows(pair_rows)


def negation_breakdown(probes: list[Probe], pred_dir: Path, out_dir: Path) -> None:
    print("\n=== NEGATION PER-ITEM BREAKDOWN ===")
    item_stats: dict[str, dict] = defaultdict(
        lambda: {"neg_correct": 0, "neg_total": 0,
                 "pos_correct": 0, "pos_total": 0,
                 "neg_captures": 0})

    neg_probes = [p for p in probes if p.phenomenon in ("negation", "negation_positive")]

    for probe in neg_probes:
        pred = _load_pred(pred_dir, probe.probe_id)
        if pred is None:
            continue
        item = _extract_negation_item(probe.prompt)
        if item is None:
            continue

        stats = item_stats[item]
        failure = _classify_failure(probe, pred)

        if probe.phenomenon == "negation":
            stats["neg_total"] += 1
            if failure == "correct":
                stats["neg_correct"] += 1
            elif failure == "distractor-capture":
                stats["neg_captures"] += 1
        else:
            stats["pos_total"] += 1
            if failure == "correct":
                stats["pos_correct"] += 1

    rows = []
    for item in sorted(item_stats.keys()):
        s = item_stats[item]
        neg_acc = s["neg_correct"] / s["neg_total"] * 100 if s["neg_total"] > 0 else 0
        pos_acc = s["pos_correct"] / s["pos_total"] * 100 if s["pos_total"] > 0 else 0
        rows.append((item, s["neg_correct"], s["neg_total"], neg_acc,
                      s["pos_correct"], s["pos_total"], pos_acc, s["neg_captures"]))
        print(f"  {item:12s}  neg={s['neg_correct']:2d}/{s['neg_total']:2d} ({neg_acc:5.1f}%)  "
              f"pos={s['pos_correct']:2d}/{s['pos_total']:2d} ({pos_acc:5.1f}%)  "
              f"captures={s['neg_captures']}")

    out_path = out_dir / "negation_per_item.csv"
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item", "neg_correct", "neg_total", "neg_accuracy",
                     "pos_correct", "pos_total", "pos_accuracy", "neg_distractor_capture"])
        w.writerows(rows)


def finegrained_breakdown(probes: list[Probe], pred_dir: Path, out_dir: Path) -> None:
    print("\n=== FINE-GRAINED PER-GROUP BREAKDOWN ===")
    group_stats: dict[str, dict] = defaultdict(
        lambda: {"correct": 0, "total": 0, "captures": 0, "other": 0, "tier": ""})

    fg_probes = [p for p in probes
                 if p.phenomenon in ("finegrained_confusable", "finegrained_distinct")]

    for probe in fg_probes:
        pred = _load_pred(pred_dir, probe.probe_id)
        if pred is None:
            continue
        group = probe.notes if probe.notes else "unknown"
        stats = group_stats[group]
        stats["tier"] = probe.phenomenon.replace("finegrained_", "")
        stats["total"] += 1
        failure = _classify_failure(probe, pred)
        if failure == "correct":
            stats["correct"] += 1
        elif failure == "distractor-capture":
            stats["captures"] += 1
        elif failure == "other-grounding":
            stats["other"] += 1

    rows = []
    for group in sorted(group_stats.keys()):
        s = group_stats[group]
        acc = s["correct"] / s["total"] * 100 if s["total"] > 0 else 0
        rows.append((group, s["tier"], s["correct"], s["total"], acc,
                      s["captures"], s["other"]))
        print(f"  {group:20s} [{s['tier']:10s}]  {s['correct']:2d}/{s['total']:2d} = {acc:5.1f}%  "
              f"captures={s['captures']}  other={s['other']}")

    out_path = out_dir / "finegrained_per_group.csv"
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "tier", "correct", "total", "accuracy",
                     "distractor_capture", "other_grounding"])
        w.writerows(rows)


def spatial_other_grounding_check(probes: list[Probe], pred_dir: Path, out_dir: Path) -> None:
    print("\n=== SPATIAL OTHER-GROUNDING SPOT-CHECK ===")
    print("  (What did the model grab instead of target or distractor?)\n")

    spatial_probes = [p for p in probes
                      if p.phenomenon in ("spatial_left", "spatial_right")]

    og_cases = []
    for probe in spatial_probes:
        pred = _load_pred(pred_dir, probe.probe_id)
        if pred is None:
            continue
        failure = _classify_failure(probe, pred)
        if failure != "other-grounding":
            continue
        og_cases.append((probe, pred))

    print(f"  Total spatial other-grounding failures: {len(og_cases)}")

    # check what was predicted: is it overlapping with known objects?
    landmark_captures = 0
    rows = []
    for probe, pred in og_cases:
        top_box = pred["top_box"]
        target_iou = box_iou(probe.target_box, top_box)
        dist_iou = box_iou(probe.distractor_box, top_box) if probe.distractor_box else 0.0

        # extract the landmark object from the prompt
        # "the X to the left/right of the Y" → landmark is Y
        m = re.search(r"(?:to the (?:left|right) of )(the .+)$", probe.prompt)
        landmark = m.group(1) if m else "?"

        rows.append({
            "probe_id": probe.probe_id,
            "prompt": probe.prompt,
            "phenomenon": probe.phenomenon,
            "landmark": landmark,
            "target_iou": round(target_iou, 3),
            "distractor_iou": round(dist_iou, 3),
            "confidence": pred.get("top_confidence", 0),
            "n_candidates": len(pred.get("candidates", [])),
        })

        # heuristic: if target_iou and dist_iou are both very low,
        # the model likely grabbed something else entirely (possibly the landmark)
        if target_iou < 0.1 and dist_iou < 0.1:
            landmark_captures += 1

    print(f"  Low IoU with BOTH target and distractor (<0.1): {landmark_captures}/{len(og_cases)}")
    print(f"  (Likely landmark captures or completely unrelated objects)\n")

    # print sample
    sample_size = min(10, len(rows))
    print(f"  Sample of {sample_size} other-grounding failures:")
    for r in rows[:sample_size]:
        print(f"    {r['probe_id']:15s} | \"{r['prompt']}\"")
        print(f"      target_iou={r['target_iou']:.3f}  dist_iou={r['distractor_iou']:.3f}  "
              f"conf={r['confidence']:.3f}  candidates={r['n_candidates']}  "
              f"landmark={r['landmark']}")

    out_path = out_dir / "spatial_other_grounding.csv"
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["probe_id", "prompt", "phenomenon", "landmark",
                                           "target_iou", "distractor_iou", "confidence",
                                           "n_candidates"])
        w.writeheader()
        w.writerows(rows)

    print(f"\n  Full table saved → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions-dir", type=Path, default=Path("predictions/grounded_sam"))
    parser.add_argument("--probes", type=Path, default=Path("probes/probe_set_v1.json"))
    parser.add_argument("--controls", type=Path, default=Path("probes/control_set_v1.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/appendix"))
    args = parser.parse_args()

    all_probes = load_probes(args.probes)
    if args.controls.exists():
        all_probes += load_probes(args.controls)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    attribute_breakdown(all_probes, args.predictions_dir, args.output_dir)
    negation_breakdown(all_probes, args.predictions_dir, args.output_dir)
    finegrained_breakdown(all_probes, args.predictions_dir, args.output_dir)
    spatial_other_grounding_check(all_probes, args.predictions_dir, args.output_dir)


if __name__ == "__main__":
    main()

"""Analyze model predictions against frozen probes.

Metrics
-------
- **Accuracy**: fraction of probes where the model's predicted mask/box
  hits the target (IoU ≥ threshold).  All verified probes count, including
  singletons whose mirror was rejected at verification.

- **Pair-consistency**: fraction of *complete* mirror pairs where the model
  gets *both* probes correct.  Only pairs where both mirrors survived
  verification are included.  This is the core distractor diagnostic.

Pair rejection policy
---------------------
When one mirror of a pair is kept but the other rejected during human
verification, the valid singleton is retained for the accuracy analysis.
Pair-consistency analysis filters to complete pairs only (both mirrors
present in the verified probe set).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.geometry import box_iou
from src.schema import Probe, load_probes


# ── Metric helpers ─────────────────────────────────────────────────

def compute_accuracy(
    probes: list[Probe],
    predictions: dict[str, Any],
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute per-probe accuracy.

    Parameters
    ----------
    probes : list[Probe]
        All verified probes (singletons + complete pairs).
    predictions : dict[str, Any]
        Mapping probe_id → {"predicted_box": [x1, y1, x2, y2], ...}.
    iou_threshold : float
        IoU threshold for a correct prediction.

    Returns
    -------
    dict with keys: total, correct, accuracy, per_phenomenon.
    """
    correct = 0
    total = 0
    per_phenomenon: dict[str, dict[str, int]] = defaultdict(
        lambda: {"correct": 0, "total": 0})

    for probe in probes:
        pred = predictions.get(probe.probe_id)
        if pred is None:
            continue
        total += 1
        pred_box = tuple(pred["predicted_box"])
        hit = box_iou(probe.target_box, pred_box) >= iou_threshold
        if hit:
            correct += 1
        per_phenomenon[probe.phenomenon]["total"] += 1
        if hit:
            per_phenomenon[probe.phenomenon]["correct"] += 1

    acc = correct / total if total > 0 else 0.0
    phenom_acc = {}
    for phenom, counts in sorted(per_phenomenon.items()):
        t, c = counts["total"], counts["correct"]
        phenom_acc[phenom] = {
            "total": t, "correct": c, "accuracy": c / t if t > 0 else 0.0,
        }

    return {"total": total, "correct": correct, "accuracy": acc,
            "per_phenomenon": phenom_acc}


def compute_pair_consistency(
    probes: list[Probe],
    predictions: dict[str, Any],
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute pair-consistency on complete mirror pairs only.

    A pair is "consistent-correct" when the model gets BOTH mirrors right.
    Only pairs where both mirrors are present in `probes` are evaluated
    (the pair rejection policy: singletons whose mirror was rejected at
    verification are excluded from this metric).

    Returns
    -------
    dict with keys: total_pairs, both_correct, consistency, per_phenomenon.
    """
    # Group probes by pair_id (skip probes without pair_id)
    by_pair: dict[str, list[Probe]] = defaultdict(list)
    for probe in probes:
        if probe.pair_id:
            by_pair[probe.pair_id].append(probe)

    # Only keep complete pairs (exactly 2 mirrors)
    complete_pairs = {pid: ps for pid, ps in by_pair.items() if len(ps) == 2}

    both_correct = 0
    total_pairs = 0
    per_phenomenon: dict[str, dict[str, int]] = defaultdict(
        lambda: {"pairs": 0, "both_correct": 0})

    for pair_id, pair_probes in sorted(complete_pairs.items()):
        hits = []
        for probe in pair_probes:
            pred = predictions.get(probe.probe_id)
            if pred is None:
                hits.append(False)
                continue
            pred_box = tuple(pred["predicted_box"])
            hits.append(box_iou(probe.target_box, pred_box) >= iou_threshold)

        total_pairs += 1
        phenom = pair_probes[0].phenomenon
        per_phenomenon[phenom]["pairs"] += 1
        if all(hits):
            both_correct += 1
            per_phenomenon[phenom]["both_correct"] += 1

    consistency = both_correct / total_pairs if total_pairs > 0 else 0.0
    phenom_cons = {}
    for phenom, counts in sorted(per_phenomenon.items()):
        p, bc = counts["pairs"], counts["both_correct"]
        phenom_cons[phenom] = {
            "pairs": p, "both_correct": bc,
            "consistency": bc / p if p > 0 else 0.0,
        }

    return {"total_pairs": total_pairs, "both_correct": both_correct,
            "consistency": consistency, "per_phenomenon": phenom_cons}


# ── Full report ────────────────────────────────────────────────────

def analyze(
    probes: list[Probe],
    predictions: dict[str, Any],
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Run accuracy + pair-consistency and return combined report."""
    acc = compute_accuracy(probes, predictions, iou_threshold)
    pair = compute_pair_consistency(probes, predictions, iou_threshold)
    return {"iou_threshold": iou_threshold, "accuracy": acc,
            "pair_consistency": pair}


def _print_report(report: dict[str, Any]) -> None:
    """Pretty-print analysis report."""
    acc = report["accuracy"]
    pair = report["pair_consistency"]
    iou_t = report["iou_threshold"]

    print(f"\n{'='*60}")
    print(f"  Probe Analysis Report  (IoU threshold = {iou_t})")
    print(f"{'='*60}")

    print(f"\n  Accuracy: {acc['correct']}/{acc['total']} "
          f"= {acc['accuracy']:.1%}")
    print(f"  (includes singletons whose mirror was rejected)")
    for phenom, pa in acc["per_phenomenon"].items():
        print(f"    {phenom:<30} {pa['correct']}/{pa['total']} "
              f"= {pa['accuracy']:.1%}")

    print(f"\n  Pair consistency: {pair['both_correct']}/{pair['total_pairs']} "
          f"= {pair['consistency']:.1%}")
    print(f"  (complete pairs only — singletons excluded)")
    for phenom, pc in pair["per_phenomenon"].items():
        print(f"    {phenom:<30} {pc['both_correct']}/{pc['pairs']} "
              f"= {pc['consistency']:.1%}")

    print(f"\n{'='*60}\n")


# ── CLI ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probes_json", type=Path,
                        help="Path to verified probes JSON")
    parser.add_argument("predictions_json", type=Path,
                        help="JSON mapping probe_id → {predicted_box: [...]}")
    parser.add_argument("--iou-threshold", type=float, default=0.5,
                        help="IoU threshold for correct prediction (default 0.5)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Save report JSON to this path")
    args = parser.parse_args()

    probes = load_probes(args.probes_json)
    with args.predictions_json.open() as f:
        predictions = json.load(f)

    report = analyze(probes, predictions, args.iou_threshold)
    _print_report(report)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as f:
            json.dump(report, f, indent=2)
        print(f"Report saved → {args.output}")


if __name__ == "__main__":
    main()

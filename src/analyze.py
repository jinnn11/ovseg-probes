"""Analyze model predictions against frozen probes.

Metrics
-------
- Selection accuracy: IoU>0.5 with target AND (if distractor) <0.5 with distractor
- Distractor gap: control accuracy − distractor accuracy per phenomenon
- Mask mIoU: where ground-truth masks exist
- Failure classification: distractor-capture / other-grounding / detection-miss /
  segmentation-error
- Pair-consistency: fraction of intact pairs where both mirrors correct
- Negation-vs-positive contrast
- Threshold sweep: accuracy at confidence cutoffs 0.15–0.5

Outputs: CSVs + matplotlib figures.

Usage:
    python -m src.analyze --predictions-dir predictions/mock \\
        --probes probes/probe_set_v1.json --controls probes/control_set_v1.json \\
        --output-dir results/mock
"""

from __future__ import annotations

import argparse
import csv
import json
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

from src.geometry import box_iou
from src.schema import Probe, load_probes


# ── Helpers ───────────────────────────────────────────────────────

def _load_prediction(pred_dir: Path, probe_id: str) -> dict | None:
    path = pred_dir / f"{probe_id}.json"
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def _mask_iou(rle_a: dict, rle_b: dict) -> float:
    """Compute mask IoU from two COCO-format RLE dicts.

    Uses pycocotools if available, otherwise falls back to a simple
    decoded-array comparison.
    """
    try:
        import pycocotools.mask as mask_util
        iou = mask_util.iou([rle_a], [rle_b], [0])
        return float(iou[0][0])
    except (ImportError, Exception):
        return _mask_iou_fallback(rle_a, rle_b)


def _decode_rle_counts(rle_str: str) -> list[int]:
    """Decode COCO compressed RLE string to run-length counts."""
    counts = []
    i = 0
    while i < len(rle_str):
        val = 0
        shift = 0
        more = True
        while more:
            c = ord(rle_str[i]) - 48
            i += 1
            val |= (c & 0x1F) << shift
            shift += 5
            more = (c & 0x20) != 0
        counts.append(val)
    return counts


def _rle_to_binary(rle: dict) -> list[int]:
    """Decode RLE to flat binary array (column-major)."""
    h, w = rle["size"]
    total = h * w
    counts_str = rle["counts"]
    if isinstance(counts_str, str):
        counts = _decode_rle_counts(counts_str)
    elif isinstance(counts_str, list):
        counts = counts_str
    else:
        return [0] * total

    binary = []
    val = 0
    for cnt in counts:
        binary.extend([val] * cnt)
        val = 1 - val
    while len(binary) < total:
        binary.append(0)
    return binary[:total]


def _mask_iou_fallback(rle_a: dict, rle_b: dict) -> float:
    a = _rle_to_binary(rle_a)
    b = _rle_to_binary(rle_b)
    if len(a) != len(b):
        return 0.0
    intersection = sum(x & y for x, y in zip(a, b))
    union = sum(x | y for x, y in zip(a, b))
    if union == 0:
        return 0.0
    return intersection / union


# ── Core metric functions ────────────────────────────────────────

def _is_correct(
    probe: Probe, pred: dict, iou_threshold: float = 0.5,
) -> bool:
    """Check if prediction is correct: IoU>threshold with target,
    and if distractor exists, IoU<threshold with distractor."""
    top_box = pred.get("top_box")
    if top_box is None:
        return False
    target_iou = box_iou(probe.target_box, top_box)
    if target_iou < iou_threshold:
        return False
    if probe.distractor_box is not None:
        dist_iou = box_iou(probe.distractor_box, top_box)
        if dist_iou >= iou_threshold:
            return False
    return True


def _classify_failure(
    probe: Probe, pred: dict, iou_threshold: float = 0.5,
    mask_threshold: float = 0.75,
) -> str:
    """Classify why a prediction failed."""
    top_box = pred.get("top_box")

    if top_box is None:
        return "detection-miss"

    target_iou = box_iou(probe.target_box, top_box)

    if probe.distractor_box is not None:
        dist_iou = box_iou(probe.distractor_box, top_box)
        if dist_iou >= iou_threshold:
            return "distractor-capture"

    if target_iou < iou_threshold:
        return "other-grounding"

    # Box is correct but check mask quality
    if (probe.target_mask is not None and pred.get("top_mask") is not None):
        m_iou = _mask_iou(probe.target_mask, pred["top_mask"])
        if m_iou < mask_threshold:
            return "segmentation-error"

    return "other-grounding"


# ── Analysis entry point ─────────────────────────────────────────

def analyze(
    probes: list[Probe],
    controls: list[Probe],
    pred_dir: Path,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Run full analysis suite. Returns a report dict."""

    all_probes = probes + controls
    results: dict[str, dict] = {}
    missing_preds = 0

    for p in all_probes:
        pred = _load_prediction(pred_dir, p.probe_id)
        if pred is None:
            missing_preds += 1
            continue
        correct = _is_correct(p, pred, iou_threshold)
        failure = None if correct else _classify_failure(p, pred, iou_threshold)

        mask_iou_val = None
        if p.target_mask is not None and pred.get("top_mask") is not None:
            mask_iou_val = _mask_iou(p.target_mask, pred["top_mask"])

        results[p.probe_id] = {
            "correct": correct,
            "failure_type": failure,
            "mask_iou": mask_iou_val,
            "top_confidence": pred.get("top_confidence"),
            "n_candidates": len(pred.get("candidates", [])),
        }

    if missing_preds:
        warnings.warn(f"{missing_preds} probes have no prediction file")

    # ── Selection accuracy ────────────────────────────────────────
    accuracy = _compute_accuracy(probes, controls, results)

    # ── Distractor gap ────────────────────────────────────────────
    gap = _compute_distractor_gap(accuracy)

    # ── Mask mIoU ─────────────────────────────────────────────────
    mask_miou = _compute_mask_miou(all_probes, results)

    # ── Failure classification ────────────────────────────────────
    failures = _compute_failures(probes, results)

    # ── Pair consistency ──────────────────────────────────────────
    pair_cons = _compute_pair_consistency(probes, results)

    # ── Negation contrast ─────────────────────────────────────────
    neg_contrast = _compute_negation_contrast(probes, results)

    # ── Threshold sweep ───────────────────────────────────────────
    sweep = _compute_threshold_sweep(all_probes, pred_dir, iou_threshold)

    return {
        "iou_threshold": iou_threshold,
        "total_probes": len(all_probes),
        "missing_predictions": missing_preds,
        "accuracy": accuracy,
        "distractor_gap": gap,
        "mask_miou": mask_miou,
        "failures": failures,
        "pair_consistency": pair_cons,
        "negation_contrast": neg_contrast,
        "threshold_sweep": sweep,
    }


def _compute_accuracy(
    probes: list[Probe], controls: list[Probe], results: dict,
) -> dict:
    """Selection accuracy per phenomenon."""
    by_phenom: dict[str, dict[str, int]] = defaultdict(
        lambda: {"correct": 0, "total": 0})

    for p in probes:
        r = results.get(p.probe_id)
        if r is None:
            continue
        by_phenom[p.phenomenon]["total"] += 1
        if r["correct"]:
            by_phenom[p.phenomenon]["correct"] += 1

    # Controls: group by base phenomenon (strip _control suffix)
    ctrl_by_phenom: dict[str, dict[str, int]] = defaultdict(
        lambda: {"correct": 0, "total": 0})
    for p in controls:
        r = results.get(p.probe_id)
        if r is None:
            continue
        ctrl_by_phenom[p.phenomenon]["total"] += 1
        if r["correct"]:
            ctrl_by_phenom[p.phenomenon]["correct"] += 1

    def _acc(d: dict) -> float:
        return d["correct"] / d["total"] if d["total"] > 0 else 0.0

    out: dict[str, Any] = {"distractor": {}, "control": {}}
    for ph in sorted(by_phenom):
        out["distractor"][ph] = {**by_phenom[ph], "accuracy": _acc(by_phenom[ph])}
    for ph in sorted(ctrl_by_phenom):
        out["control"][ph] = {**ctrl_by_phenom[ph], "accuracy": _acc(ctrl_by_phenom[ph])}

    # Aggregate
    d_total = sum(v["total"] for v in by_phenom.values())
    d_correct = sum(v["correct"] for v in by_phenom.values())
    c_total = sum(v["total"] for v in ctrl_by_phenom.values())
    c_correct = sum(v["correct"] for v in ctrl_by_phenom.values())
    out["distractor_overall"] = {
        "correct": d_correct, "total": d_total,
        "accuracy": d_correct / d_total if d_total else 0.0}
    out["control_overall"] = {
        "correct": c_correct, "total": c_total,
        "accuracy": c_correct / c_total if c_total else 0.0}

    return out


PHENOMENON_TO_CONTROL = {
    "attribute_color": "attribute_color_control",
    "spatial_left": "spatial_left_control",
    "spatial_right": "spatial_right_control",
    "finegrained_confusable": "finegrained_control",
    "finegrained_distinct": "finegrained_control",
    "negation": "negation_control",
    "negation_positive": "negation_control",
}


def _compute_distractor_gap(accuracy: dict) -> dict:
    """Control accuracy − distractor accuracy per phenomenon."""
    gaps = {}
    for phenom, ctrl_phenom in PHENOMENON_TO_CONTROL.items():
        d = accuracy["distractor"].get(phenom)
        c = accuracy["control"].get(ctrl_phenom)
        if d and c:
            gaps[phenom] = {
                "control_acc": c["accuracy"],
                "distractor_acc": d["accuracy"],
                "gap": c["accuracy"] - d["accuracy"],
            }
    return gaps


def _compute_mask_miou(probes: list[Probe], results: dict) -> dict:
    """Mean mask IoU where ground-truth masks exist."""
    by_phenom: dict[str, list[float]] = defaultdict(list)
    for p in probes:
        r = results.get(p.probe_id)
        if r is None or r["mask_iou"] is None:
            continue
        by_phenom[p.phenomenon].append(r["mask_iou"])

    out = {}
    all_ious: list[float] = []
    for ph in sorted(by_phenom):
        vals = by_phenom[ph]
        out[ph] = {"mean_iou": sum(vals) / len(vals), "count": len(vals)}
        all_ious.extend(vals)
    if all_ious:
        out["overall"] = {"mean_iou": sum(all_ious) / len(all_ious),
                          "count": len(all_ious)}
    return out


def _compute_failures(probes: list[Probe], results: dict) -> dict:
    """Failure classification counts per phenomenon."""
    by_phenom: dict[str, Counter] = defaultdict(Counter)
    for p in probes:
        r = results.get(p.probe_id)
        if r is None or r["correct"]:
            continue
        by_phenom[p.phenomenon][r["failure_type"]] += 1

    out = {}
    for ph in sorted(by_phenom):
        out[ph] = dict(by_phenom[ph].most_common())
    overall = Counter()
    for c in by_phenom.values():
        overall.update(c)
    out["overall"] = dict(overall.most_common())
    return out


def _compute_pair_consistency(probes: list[Probe], results: dict) -> dict:
    """Pair-consistency: both mirrors correct in intact pairs."""
    by_pair: dict[str, list[Probe]] = defaultdict(list)
    for p in probes:
        if p.pair_id:
            by_pair[p.pair_id].append(p)

    complete_pairs = {pid: ps for pid, ps in by_pair.items() if len(ps) == 2}

    by_phenom: dict[str, dict[str, int]] = defaultdict(
        lambda: {"pairs": 0, "both_correct": 0})

    for pair_id, pair_probes in sorted(complete_pairs.items()):
        hits = []
        for p in pair_probes:
            r = results.get(p.probe_id)
            hits.append(r is not None and r["correct"])

        phenom = pair_probes[0].phenomenon
        by_phenom[phenom]["pairs"] += 1
        if all(hits):
            by_phenom[phenom]["both_correct"] += 1

    out = {}
    total_pairs = 0
    total_bc = 0
    for ph in sorted(by_phenom):
        d = by_phenom[ph]
        out[ph] = {**d, "consistency": d["both_correct"] / d["pairs"]
                   if d["pairs"] else 0.0}
        total_pairs += d["pairs"]
        total_bc += d["both_correct"]
    out["overall"] = {
        "pairs": total_pairs, "both_correct": total_bc,
        "consistency": total_bc / total_pairs if total_pairs else 0.0}
    return out


def _compute_negation_contrast(probes: list[Probe], results: dict) -> dict:
    """Negation vs negation_positive accuracy, plus within intact pairs."""
    neg_correct = neg_total = 0
    pos_correct = pos_total = 0

    for p in probes:
        r = results.get(p.probe_id)
        if r is None:
            continue
        if p.phenomenon == "negation":
            neg_total += 1
            if r["correct"]:
                neg_correct += 1
        elif p.phenomenon == "negation_positive":
            pos_total += 1
            if r["correct"]:
                pos_correct += 1

    # Within intact negation pairs
    by_pair: dict[str, list[Probe]] = defaultdict(list)
    for p in probes:
        if p.pair_id and p.phenomenon in ("negation", "negation_positive"):
            by_pair[p.pair_id].append(p)

    neg_pairs = {pid: ps for pid, ps in by_pair.items() if len(ps) == 2}
    pair_both = 0
    pair_neg_only = 0
    pair_pos_only = 0
    pair_neither = 0

    for pair_probes in neg_pairs.values():
        neg_p = [p for p in pair_probes if p.phenomenon == "negation"]
        pos_p = [p for p in pair_probes if p.phenomenon == "negation_positive"]
        if not neg_p or not pos_p:
            continue
        nr = results.get(neg_p[0].probe_id)
        pr = results.get(pos_p[0].probe_id)
        n_hit = nr is not None and nr["correct"]
        p_hit = pr is not None and pr["correct"]
        if n_hit and p_hit:
            pair_both += 1
        elif n_hit:
            pair_neg_only += 1
        elif p_hit:
            pair_pos_only += 1
        else:
            pair_neither += 1

    return {
        "negation": {"correct": neg_correct, "total": neg_total,
                     "accuracy": neg_correct / neg_total if neg_total else 0.0},
        "negation_positive": {"correct": pos_correct, "total": pos_total,
                              "accuracy": pos_correct / pos_total if pos_total else 0.0},
        "intact_pairs": {
            "total": len(neg_pairs),
            "both_correct": pair_both,
            "neg_only": pair_neg_only,
            "pos_only": pair_pos_only,
            "neither": pair_neither,
        },
    }


def _compute_threshold_sweep(
    probes: list[Probe], pred_dir: Path, iou_threshold: float,
) -> list[dict]:
    """Recompute accuracy at confidence cutoffs 0.15–0.50."""
    cutoffs = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    sweep = []

    for cutoff in cutoffs:
        correct = 0
        total = 0
        for p in probes:
            pred = _load_prediction(pred_dir, p.probe_id)
            if pred is None:
                continue
            top_conf = pred.get("top_confidence")
            if top_conf is not None and top_conf < cutoff:
                continue
            total += 1
            if _is_correct(p, pred, iou_threshold):
                correct += 1
        sweep.append({
            "confidence_cutoff": cutoff,
            "evaluated": total,
            "correct": correct,
            "accuracy": correct / total if total > 0 else 0.0,
        })
    return sweep


# ── Output: CSVs + Figures ────────────────────────────────────────

def _save_csvs(report: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Accuracy CSV
    with (out_dir / "accuracy.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["type", "phenomenon", "correct", "total", "accuracy"])
        for ph, d in report["accuracy"]["distractor"].items():
            w.writerow(["distractor", ph, d["correct"], d["total"],
                        f"{d['accuracy']:.4f}"])
        for ph, d in report["accuracy"]["control"].items():
            w.writerow(["control", ph, d["correct"], d["total"],
                        f"{d['accuracy']:.4f}"])

    # Distractor gap CSV
    with (out_dir / "distractor_gap.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["phenomenon", "control_acc", "distractor_acc", "gap"])
        for ph, d in report["distractor_gap"].items():
            w.writerow([ph, f"{d['control_acc']:.4f}",
                        f"{d['distractor_acc']:.4f}", f"{d['gap']:.4f}"])

    # Failures CSV
    with (out_dir / "failures.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["phenomenon", "failure_type", "count"])
        for ph, types in report["failures"].items():
            for ft, cnt in types.items():
                w.writerow([ph, ft, cnt])

    # Pair consistency CSV
    with (out_dir / "pair_consistency.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["phenomenon", "pairs", "both_correct", "consistency"])
        for ph, d in report["pair_consistency"].items():
            w.writerow([ph, d["pairs"], d["both_correct"],
                        f"{d['consistency']:.4f}"])

    # Threshold sweep CSV
    with (out_dir / "threshold_sweep.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["confidence_cutoff", "evaluated", "correct", "accuracy"])
        for s in report["threshold_sweep"]:
            w.writerow([s["confidence_cutoff"], s["evaluated"],
                        s["correct"], f"{s['accuracy']:.4f}"])

    # Mask mIoU CSV
    if report["mask_miou"]:
        with (out_dir / "mask_miou.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["phenomenon", "mean_iou", "count"])
            for ph, d in report["mask_miou"].items():
                w.writerow([ph, f"{d['mean_iou']:.4f}", d["count"]])


def _save_figures(report: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _fig_accuracy_bars(report, out_dir)
    _fig_failure_bars(report, out_dir)
    _fig_gap_chart(report, out_dir)
    _fig_threshold_sweep(report, out_dir)


def _fig_accuracy_bars(report: dict, out_dir: Path) -> None:
    dist = report["accuracy"]["distractor"]
    ctrl = report["accuracy"]["control"]

    # Merge phenomena
    all_phenom = sorted(set(list(dist.keys()) + list(ctrl.keys())))
    if not all_phenom:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(all_phenom))
    width = 0.35

    dist_vals = [dist.get(ph, {}).get("accuracy", 0) * 100 for ph in all_phenom]
    ctrl_vals = [ctrl.get(ph, {}).get("accuracy", 0) * 100 for ph in all_phenom]

    bars1 = ax.bar([i - width / 2 for i in x], dist_vals, width, label="Distractor")
    bars2 = ax.bar([i + width / 2 for i in x], ctrl_vals, width, label="Control")

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Selection Accuracy by Phenomenon")
    ax.set_xticks(list(x))
    ax.set_xticklabels(all_phenom, rotation=45, ha="right", fontsize=8)
    ax.legend()
    ax.set_ylim(0, 105)

    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1,
                f"{h:.0f}", ha="center", fontsize=7)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1,
                f"{h:.0f}", ha="center", fontsize=7)

    plt.tight_layout()
    fig.savefig(out_dir / "accuracy_bars.png", dpi=150)
    plt.close(fig)


def _fig_failure_bars(report: dict, out_dir: Path) -> None:
    failures = {k: v for k, v in report["failures"].items() if k != "overall"}
    if not failures:
        return

    phenoms = sorted(failures.keys())
    failure_types = ["distractor-capture", "other-grounding",
                     "detection-miss", "segmentation-error"]
    colors = ["#e74c3c", "#f39c12", "#95a5a6", "#3498db"]

    fig, ax = plt.subplots(figsize=(12, 6))
    bottom = [0.0] * len(phenoms)

    for ft, color in zip(failure_types, colors):
        vals = [failures[ph].get(ft, 0) for ph in phenoms]
        ax.bar(phenoms, vals, bottom=bottom, label=ft, color=color)
        bottom = [b + v for b, v in zip(bottom, vals)]

    ax.set_ylabel("Count")
    ax.set_title("Failure Classification by Phenomenon")
    ax.set_xticks(range(len(phenoms)))
    ax.set_xticklabels(phenoms, rotation=45, ha="right", fontsize=8)
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(out_dir / "failure_bars.png", dpi=150)
    plt.close(fig)


def _fig_gap_chart(report: dict, out_dir: Path) -> None:
    gap = report["distractor_gap"]
    if not gap:
        return

    phenoms = sorted(gap.keys())
    gaps = [gap[ph]["gap"] * 100 for ph in phenoms]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(phenoms, gaps, color=["#e74c3c" if g > 0 else "#2ecc71"
                                        for g in gaps])
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_ylabel("Distractor Gap (pp)")
    ax.set_title("Distractor Gap: Control Accuracy − Distractor Accuracy")
    ax.set_xticks(range(len(phenoms)))
    ax.set_xticklabels(phenoms, rotation=45, ha="right", fontsize=8)

    for bar, val in zip(bars, gaps):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (1 if val >= 0 else -3),
                f"{val:.1f}", ha="center", fontsize=8)

    plt.tight_layout()
    fig.savefig(out_dir / "distractor_gap.png", dpi=150)
    plt.close(fig)


def _fig_threshold_sweep(report: dict, out_dir: Path) -> None:
    sweep = report["threshold_sweep"]
    if not sweep:
        return

    cutoffs = [s["confidence_cutoff"] for s in sweep]
    accs = [s["accuracy"] * 100 for s in sweep]
    counts = [s["evaluated"] for s in sweep]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(cutoffs, accs, "b-o", label="Accuracy (%)")
    ax1.set_xlabel("Confidence Cutoff")
    ax1.set_ylabel("Accuracy (%)", color="b")
    ax1.set_ylim(0, 105)

    ax2 = ax1.twinx()
    ax2.bar(cutoffs, counts, width=0.03, alpha=0.3, color="gray",
            label="Probes evaluated")
    ax2.set_ylabel("Probes evaluated", color="gray")

    ax1.set_title("Accuracy vs Confidence Threshold")
    fig.legend(loc="upper right", bbox_to_anchor=(0.88, 0.88))
    plt.tight_layout()
    fig.savefig(out_dir / "threshold_sweep.png", dpi=150)
    plt.close(fig)


# ── Console report ────────────────────────────────────────────────

def _print_report(report: dict) -> None:
    iou_t = report["iou_threshold"]
    acc = report["accuracy"]

    print(f"\n{'='*65}")
    print(f"  Analysis Report  (IoU threshold = {iou_t})")
    print(f"  Probes: {report['total_probes']}  |  "
          f"Missing predictions: {report['missing_predictions']}")
    print(f"{'='*65}")

    print(f"\n  SELECTION ACCURACY")
    print(f"  {'Distractor':<40} "
          f"{acc['distractor_overall']['correct']}/{acc['distractor_overall']['total']} "
          f"= {acc['distractor_overall']['accuracy']:.1%}")
    for ph, d in acc["distractor"].items():
        print(f"    {ph:<38} {d['correct']}/{d['total']} = {d['accuracy']:.1%}")
    print(f"  {'Control':<40} "
          f"{acc['control_overall']['correct']}/{acc['control_overall']['total']} "
          f"= {acc['control_overall']['accuracy']:.1%}")
    for ph, d in acc["control"].items():
        print(f"    {ph:<38} {d['correct']}/{d['total']} = {d['accuracy']:.1%}")

    print(f"\n  DISTRACTOR GAP")
    for ph, d in report["distractor_gap"].items():
        print(f"    {ph:<38} {d['gap']:+.1%} "
              f"(ctrl={d['control_acc']:.1%}, dist={d['distractor_acc']:.1%})")

    if report["mask_miou"]:
        print(f"\n  MASK mIoU")
        for ph, d in report["mask_miou"].items():
            print(f"    {ph:<38} {d['mean_iou']:.3f}  (n={d['count']})")

    print(f"\n  FAILURE CLASSIFICATION")
    for ph, types in report["failures"].items():
        parts = ", ".join(f"{ft}={cnt}" for ft, cnt in types.items())
        print(f"    {ph:<38} {parts}")

    print(f"\n  PAIR CONSISTENCY")
    for ph, d in report["pair_consistency"].items():
        print(f"    {ph:<38} {d['both_correct']}/{d['pairs']} "
              f"= {d['consistency']:.1%}")

    nc = report["negation_contrast"]
    if nc["negation"]["total"] > 0:
        print(f"\n  NEGATION CONTRAST")
        print(f"    negation:          {nc['negation']['accuracy']:.1%} "
              f"({nc['negation']['correct']}/{nc['negation']['total']})")
        print(f"    negation_positive: {nc['negation_positive']['accuracy']:.1%} "
              f"({nc['negation_positive']['correct']}/{nc['negation_positive']['total']})")
        ip = nc["intact_pairs"]
        print(f"    intact pairs: {ip['total']}  "
              f"(both={ip['both_correct']}, neg_only={ip['neg_only']}, "
              f"pos_only={ip['pos_only']}, neither={ip['neither']})")

    print(f"\n  THRESHOLD SWEEP")
    for s in report["threshold_sweep"]:
        print(f"    cutoff={s['confidence_cutoff']:.2f}  "
              f"acc={s['accuracy']:.1%}  (n={s['evaluated']})")

    print(f"\n{'='*65}\n")


# ── CLI ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions-dir", required=True, type=Path)
    parser.add_argument("--probes", required=True, type=Path)
    parser.add_argument("--controls", required=True, type=Path)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    probes = load_probes(args.probes)
    controls = load_probes(args.controls)

    report = analyze(probes, controls, args.predictions_dir, args.iou_threshold)
    _print_report(report)

    if args.output_dir:
        _save_csvs(report, args.output_dir)
        _save_figures(report, args.output_dir)
        # Save full report JSON
        report_path = args.output_dir / "report.json"
        with report_path.open("w") as f:
            json.dump(report, f, indent=2)
        print(f"Results saved → {args.output_dir}/")


if __name__ == "__main__":
    main()

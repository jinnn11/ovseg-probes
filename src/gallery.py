"""Render failure gallery images for all distractor-capture failures.

Each image shows:
  - Predicted box/mask in RED
  - GT target box in GREEN dashed
  - GT distractor box in ORANGE dashed
  - Prompt as title, confidence + probe_id as subtitle

Usage:
    python -m src.gallery --predictions-dir predictions/grounded_sam
    python -m src.gallery --predictions-dir predictions/grounded_sam --failure-type all
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

from src.geometry import box_iou
from src.schema import Probe, load_probes


IMAGES_DIR = Path("data/images")


def _get_image_path(probe: Probe) -> Path | None:
    if probe.image_source in ("coco_train2017", "lvis_v1_train"):
        return IMAGES_DIR / "coco" / f"{int(probe.image_id):012d}.jpg"
    elif probe.image_source == "visual_genome":
        vg_dir = IMAGES_DIR / "vg"
        for ext in ("jpg", "png", "jpeg"):
            p = vg_dir / f"{probe.image_id}.{ext}"
            if p.exists():
                return p
    return None


def _decode_rle(rle: dict) -> np.ndarray | None:
    try:
        import pycocotools.mask as mask_util
        if isinstance(rle["counts"], str):
            mask = mask_util.decode(rle)
        else:
            mask = mask_util.decode(mask_util.frPyObjects(rle, *rle["size"]))
        return mask.squeeze()
    except Exception:
        return None


def _classify_failure(probe: Probe, pred: dict, iou_threshold: float = 0.5) -> str:
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
    return "correct"


def _is_correct(probe: Probe, pred: dict, iou_threshold: float = 0.5) -> bool:
    return _classify_failure(probe, pred, iou_threshold) == "correct"


def _draw_box(ax, box, color, linewidth=2, linestyle="-", label=None):
    x1, y1, x2, y2 = box
    rect = patches.Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        linewidth=linewidth, edgecolor=color,
        facecolor="none", linestyle=linestyle, label=label,
    )
    ax.add_patch(rect)


def render_failure(probe: Probe, pred: dict, ax: plt.Axes) -> None:
    img_path = _get_image_path(probe)
    if img_path is None or not img_path.exists():
        ax.text(0.5, 0.5, f"Image not found\n{probe.probe_id}",
                ha="center", va="center", transform=ax.transAxes)
        return

    img = Image.open(img_path)
    img_array = np.array(img)
    ax.imshow(img_array)

    top_mask_rle = pred.get("top_mask")
    if top_mask_rle is not None:
        mask = _decode_rle(top_mask_rle)
        if mask is not None and mask.shape == img_array.shape[:2]:
            overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
            overlay[mask > 0] = [1.0, 0.2, 0.2, 0.35]
            ax.imshow(overlay)

    _draw_box(ax, probe.target_box, "lime", linewidth=2, linestyle="--",
              label="GT target")

    if probe.distractor_box is not None:
        _draw_box(ax, probe.distractor_box, "orange", linewidth=2,
                  linestyle="--", label="GT distractor")

    top_box = pred.get("top_box")
    if top_box is not None:
        _draw_box(ax, top_box, "red", linewidth=3, linestyle="-",
                  label="Predicted")

    conf = pred.get("top_confidence") or 0.0
    failure = _classify_failure(probe, pred)
    ax.set_title(f'"{probe.prompt}"', fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel(
        f"{probe.probe_id} | {probe.phenomenon} | "
        f"conf={conf:.3f} | {failure}",
        fontsize=8, color="gray",
    )
    ax.legend(loc="upper right", fontsize=7, framealpha=0.7)
    ax.axis("off")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions-dir", type=Path, default=Path("predictions/grounded_sam"))
    parser.add_argument("--probes", type=Path, default=Path("probes/probe_set_v1.json"))
    parser.add_argument("--controls", type=Path, default=Path("probes/control_set_v1.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("gallery"))
    parser.add_argument("--failure-type", default="distractor-capture",
                        choices=["distractor-capture", "other-grounding", "detection-miss", "all"])
    args = parser.parse_args()

    pred_dir = args.predictions_dir
    all_probes = load_probes(args.probes)
    if args.controls.exists():
        all_probes += load_probes(args.controls)
    probe_map = {p.probe_id: p for p in all_probes}

    failures_by_phenom = defaultdict(list)
    all_failures = []

    for pf in sorted(pred_dir.glob("*.json")):
        with pf.open() as f:
            pred = json.load(f)
        probe_id = pred["probe_id"]
        probe = probe_map.get(probe_id)
        if probe is None:
            continue

        failure = _classify_failure(probe, pred)
        if failure == "correct":
            continue
        if args.failure_type != "all" and failure != args.failure_type:
            continue

        all_failures.append((probe, pred, failure))
        failures_by_phenom[probe.phenomenon].append((probe, pred, failure))

    if not all_failures:
        print("No failures found.")
        return

    out_dir = args.output_dir / args.failure_type.replace("-", "_")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Rendering {len(all_failures)} {args.failure_type} failures:")
    for phenom, items in sorted(failures_by_phenom.items()):
        print(f"  {phenom}: {len(items)}")

    for probe, pred, failure in all_failures:
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        render_failure(probe, pred, ax)
        fig.savefig(out_dir / f"{probe.probe_id}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

    # also render mirror pairs side-by-side where both exist in failures
    pair_dir = out_dir / "pairs"
    pair_dir.mkdir(exist_ok=True)

    pair_groups = defaultdict(list)
    for probe, pred, failure in all_failures:
        if probe.pair_id:
            pair_groups[probe.pair_id].append((probe, pred, failure))

    pair_count = 0
    for pair_id, items in sorted(pair_groups.items()):
        if len(items) < 2:
            continue
        fig, axes = plt.subplots(1, len(items), figsize=(14 * len(items), 10))
        if len(items) == 1:
            axes = [axes]
        for i, (probe, pred, failure) in enumerate(items):
            render_failure(probe, pred, axes[i])
        fig.suptitle(f"Mirror pair: {pair_id}", fontsize=14, y=1.02)
        plt.tight_layout()
        fig.savefig(pair_dir / f"{pair_id}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        pair_count += 1

    print(f"\nSaved {len(all_failures)} individual images → {out_dir}/")
    print(f"Saved {pair_count} mirror-pair composites → {pair_dir}/")


if __name__ == "__main__":
    main()

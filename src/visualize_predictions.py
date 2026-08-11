"""Visualize model predictions overlaid on images.

Renders each probe's image with:
  - Ground-truth target box (green dashed)
  - Ground-truth distractor box (orange dashed, if present)
  - Predicted box (cyan solid)
  - Predicted mask (semi-transparent blue overlay)
  - All candidate boxes (thin gray)
  - Prompt as title, confidence + probe_id as subtitle

Usage:
    python -m src.visualize_predictions --model grounded_sam --n 10
    python -m src.visualize_predictions --model grounded_sam --output-dir viz/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

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
    """Decode COCO RLE to binary mask array (H, W)."""
    try:
        import pycocotools.mask as mask_util
        if isinstance(rle["counts"], str):
            mask = mask_util.decode(rle)
        else:
            mask = mask_util.decode(mask_util.frPyObjects(rle, *rle["size"]))
        return mask.squeeze()
    except Exception:
        return _decode_rle_fallback(rle)


def _decode_rle_fallback(rle: dict) -> np.ndarray | None:
    """Fallback RLE decoder without pycocotools."""
    h, w = rle["size"]
    counts_str = rle["counts"]
    if isinstance(counts_str, str):
        counts = []
        i = 0
        while i < len(counts_str):
            val = 0
            shift = 0
            more = True
            while more:
                c = ord(counts_str[i]) - 48
                i += 1
                val |= (c & 0x1F) << shift
                shift += 5
                more = (c & 0x20) != 0
            counts.append(val)
    elif isinstance(counts_str, list):
        counts = counts_str
    else:
        return None

    binary = []
    v = 0
    for cnt in counts:
        binary.extend([v] * cnt)
        v = 1 - v
    arr = np.array(binary[:h * w], dtype=np.uint8).reshape((h, w), order="F")
    return arr


def _draw_box(ax, box, color, linewidth=2, linestyle="-", label=None):
    x1, y1, x2, y2 = box
    rect = patches.Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        linewidth=linewidth, edgecolor=color,
        facecolor="none", linestyle=linestyle, label=label,
    )
    ax.add_patch(rect)


def visualize_probe(
    probe: Probe, pred: dict, ax: plt.Axes,
) -> None:
    img_path = _get_image_path(probe)
    if img_path is None or not img_path.exists():
        ax.text(0.5, 0.5, f"Image not found\n{probe.probe_id}",
                ha="center", va="center", transform=ax.transAxes)
        return

    img = Image.open(img_path)
    img_array = np.array(img)
    ax.imshow(img_array)

    # Predicted mask overlay
    top_mask_rle = pred.get("top_mask")
    if top_mask_rle is not None:
        mask = _decode_rle(top_mask_rle)
        if mask is not None and mask.shape == img_array.shape[:2]:
            overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
            overlay[mask > 0] = [0.2, 0.5, 1.0, 0.35]
            ax.imshow(overlay)

    # All candidate boxes (thin gray)
    for i, cand in enumerate(pred.get("candidates", [])[1:], start=1):
        _draw_box(ax, cand["box_xyxy"], "gray", linewidth=1, linestyle=":")

    # GT target box (green dashed)
    _draw_box(ax, probe.target_box, "lime", linewidth=2, linestyle="--",
              label="GT target")

    # GT distractor box (orange dashed)
    if probe.distractor_box is not None:
        _draw_box(ax, probe.distractor_box, "orange", linewidth=2,
                  linestyle="--", label="GT distractor")

    # Predicted top box (cyan solid)
    top_box = pred.get("top_box")
    if top_box is not None:
        _draw_box(ax, top_box, "cyan", linewidth=3, linestyle="-",
                  label="Predicted")

    # Title
    conf = pred.get("top_confidence", 0)
    n_cands = len(pred.get("candidates", []))
    ax.set_title(f'"{probe.prompt}"', fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel(
        f"{probe.probe_id} | {probe.phenomenon} | "
        f"conf={conf:.3f} | candidates={n_cands}",
        fontsize=8, color="gray",
    )
    ax.legend(loc="upper right", fontsize=7, framealpha=0.7)
    ax.axis("off")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--n", type=int, default=10,
                        help="Number of probes to visualize")
    parser.add_argument("--predictions-dir", type=Path, default=Path("predictions"))
    parser.add_argument("--probes", type=Path, default=Path("probes/probe_set_v1.json"))
    parser.add_argument("--controls", type=Path, default=Path("probes/control_set_v1.json"))
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    pred_dir = args.predictions_dir / args.model

    # Load all probes, find ones with predictions
    all_probes = load_probes(args.probes)
    if args.controls.exists():
        all_probes += load_probes(args.controls)

    probe_map = {p.probe_id: p for p in all_probes}

    pred_files = sorted(pred_dir.glob("*.json"))[:args.n]
    if not pred_files:
        print(f"No predictions found in {pred_dir}/")
        return

    print(f"Visualizing {len(pred_files)} predictions from {pred_dir}/")

    out_dir = args.output_dir or Path(f"viz/{args.model}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Grid view
    n = len(pred_files)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 6 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, pf in enumerate(pred_files):
        with pf.open() as f:
            pred = json.load(f)
        probe_id = pred["probe_id"]
        probe = probe_map.get(probe_id)
        if probe is None:
            print(f"  Warning: {probe_id} not found in probe files")
            continue

        visualize_probe(probe, pred, axes[i])

        # Also save individual image
        fig_single, ax_single = plt.subplots(1, 1, figsize=(14, 10))
        visualize_probe(probe, pred, ax_single)
        fig_single.savefig(out_dir / f"{probe_id}.png", dpi=120,
                           bbox_inches="tight")
        plt.close(fig_single)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    grid_path = out_dir / "grid.png"
    fig.savefig(grid_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {len(pred_files)} individual images + grid → {out_dir}/")
    print(f"Grid: {grid_path}")


if __name__ == "__main__":
    main()

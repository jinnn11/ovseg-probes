"""RefCOCO sanity-check evaluation.

Session 1 gate: run Grounded-SAM on RefCOCO val split, verify the model
produces sensible results (expect ~70-80% acc@0.5) before spending GPU
time on probe inference.

RefCOCO format (from refer API or pre-extracted JSON):
  Each sample: image_id, bbox (xywh), sentence, split.

Usage:
    python -m src.eval_refcoco --model grounded_sam --refcoco-dir data/refcoco
    python -m src.eval_refcoco --model mock --refcoco-dir data/refcoco  # dry run
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from src.geometry import box_iou, xywh_to_xyxy


COCO_IMAGES_DIR = Path("data/images/coco")


def _load_refcoco(refcoco_dir: Path, split: str = "val") -> list[dict]:
    """Load RefCOCO samples from pre-extracted JSON.

    Expected format: list of dicts with keys:
      image_id, bbox (xywh), sentences (list of str), split, ann_id

    If the standard refer API pickle files exist, we load from those instead.
    """
    # Try pre-extracted JSON first
    json_path = refcoco_dir / f"refcoco_{split}.json"
    if json_path.exists():
        with json_path.open() as f:
            return json.load(f)

    # Try refer API pickle format
    return _load_from_refer_api(refcoco_dir, split)


def _load_from_refer_api(refcoco_dir: Path, split: str) -> list[dict]:
    """Load from the standard refer API pickle files."""
    import pickle

    refs_path = refcoco_dir / "refs(unc).p"
    if not refs_path.exists():
        refs_path = refcoco_dir / "refs(google).p"
    instances_path = refcoco_dir / "instances.json"

    if not refs_path.exists() or not instances_path.exists():
        raise FileNotFoundError(
            f"RefCOCO data not found in {refcoco_dir}. Expected either:\n"
            f"  - {refcoco_dir}/refcoco_{split}.json (pre-extracted), or\n"
            f"  - {refcoco_dir}/refs(unc).p + instances.json (refer API format)"
        )

    with refs_path.open("rb") as f:
        refs = pickle.load(f)

    with instances_path.open() as f:
        instances = json.load(f)

    # Build ann_id → bbox mapping
    ann_to_bbox = {ann["id"]: ann["bbox"] for ann in instances["annotations"]}

    samples = []
    for ref in refs:
        if ref["split"] != split:
            continue
        ann_id = ref["ann_id"]
        bbox = ann_to_bbox.get(ann_id)
        if bbox is None:
            continue
        sentences = [s["raw"] for s in ref["sentences"]]
        samples.append({
            "image_id": ref["image_id"],
            "ann_id": ann_id,
            "bbox": bbox,  # xywh
            "sentences": sentences,
            "split": split,
        })

    return samples


def _get_coco_image_path(image_id: int) -> Path:
    return COCO_IMAGES_DIR / f"{image_id:012d}.jpg"


def evaluate_refcoco(
    model_name: str,
    refcoco_dir: Path,
    split: str = "val",
    max_samples: int = 0,
    iou_threshold: float = 0.5,
) -> dict:
    """Run model on RefCOCO and compute accuracy.

    Parameters
    ----------
    model_name : str
        Model to use (mock, grounded_sam).
    refcoco_dir : Path
        Directory containing RefCOCO data.
    split : str
        Dataset split (val, testA, testB).
    max_samples : int
        Limit evaluation to N samples (0 = all).
    iou_threshold : float
        IoU threshold for correct prediction.
    """
    from src.run_inference import _build_model

    print(f"Loading RefCOCO {split} from {refcoco_dir} ...")
    samples = _load_refcoco(refcoco_dir, split)
    print(f"Loaded {len(samples)} samples")

    if max_samples > 0:
        samples = samples[:max_samples]
        print(f"Limited to {len(samples)} samples")

    model = _build_model(model_name)

    correct = 0
    total = 0
    detection_miss = 0
    times: list[float] = []

    for i, sample in enumerate(samples):
        image_id = sample["image_id"]
        gt_bbox_xywh = sample["bbox"]
        gt_box = xywh_to_xyxy(gt_bbox_xywh)
        sentences = sample["sentences"]

        img_path = _get_coco_image_path(image_id)
        if not img_path.exists():
            continue

        # Use first sentence as the prompt
        prompt = sentences[0]

        try:
            from PIL import Image
            with Image.open(img_path) as img:
                img_w, img_h = img.size
        except Exception:
            img_w, img_h = 640, 480

        t0 = time.time()
        detections = model.predict(str(img_path), prompt, img_w, img_h)
        elapsed = time.time() - t0
        times.append(elapsed)

        total += 1

        if not detections:
            detection_miss += 1
            continue

        # Top detection
        pred_box = detections[0].box_xyxy
        iou = box_iou(gt_box, pred_box)
        if iou >= iou_threshold:
            correct += 1

        if (i + 1) % 100 == 0:
            acc_so_far = correct / total * 100 if total > 0 else 0
            avg_time = sum(times) / len(times)
            print(f"  [{i + 1}/{len(samples)}] acc={acc_so_far:.1f}% "
                  f"miss={detection_miss} avg_time={avg_time:.3f}s")

    accuracy = correct / total if total > 0 else 0.0
    avg_time = sum(times) / len(times) if times else 0.0

    result = {
        "model": model_name,
        "split": split,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "detection_miss": detection_miss,
        "iou_threshold": iou_threshold,
        "avg_time_s": round(avg_time, 4),
    }

    print(f"\n{'='*50}")
    print(f"  RefCOCO {split} — {model_name}")
    print(f"  Accuracy@{iou_threshold}: {correct}/{total} = {accuracy:.1%}")
    print(f"  Detection miss: {detection_miss}")
    print(f"  Avg inference time: {avg_time:.3f}s")
    print(f"{'='*50}\n")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True,
                        choices=["mock", "mock_cheat", "grounded_sam"])
    parser.add_argument("--refcoco-dir", required=True, type=Path,
                        help="Directory with RefCOCO data")
    parser.add_argument("--split", default="val",
                        choices=["val", "testA", "testB"])
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Limit to N samples (0=all)")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=None,
                        help="Save result JSON")
    args = parser.parse_args()

    result = evaluate_refcoco(
        args.model, args.refcoco_dir, args.split,
        args.max_samples, args.iou_threshold,
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as f:
            json.dump(result, f, indent=2)
        print(f"Result saved → {args.output}")


if __name__ == "__main__":
    main()

"""Run model inference on frozen probes.

Produces one JSON file per probe in predictions/{model}/{probe_id}.json.
Skips probes whose output file already exists (checkpoint-safe).

Usage:
    python -m src.run_inference --model mock --probe-file probes/probe_set_v1.json
    python -m src.run_inference --model mock_cheat --probe-file probes/probe_set_v1.json
    python -m src.run_inference --model mock --probe-file probes/probe_set_v1.json --oracle-box
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

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


def _get_image_dims(probe: Probe) -> tuple[int, int]:
    """Return (width, height) for a probe's image."""
    img_path = _get_image_path(probe)
    if img_path is not None and img_path.exists():
        with Image.open(img_path) as img:
            return img.size  # (width, height)
    return (640, 480)


def _build_model(model_name: str):
    if model_name == "mock":
        from src.mock_model import MockModel
        return MockModel(seed=42)
    elif model_name == "mock_cheat":
        from src.mock_model import MockCheatModel
        return MockCheatModel(seed=42, hit_rate=0.90)
    elif model_name == "grounded_sam":
        from src.grounded_sam import GroundedSAMModel
        return GroundedSAMModel()
    else:
        raise ValueError(f"Unknown model: {model_name}")


def _build_sam(model_name: str):
    if model_name in ("mock", "mock_cheat"):
        from src.mock_model import MockSAM
        return MockSAM()
    elif model_name == "grounded_sam":
        from src.grounded_sam import GroundedSAM
        return GroundedSAM()
    else:
        raise ValueError(f"Unknown model for SAM: {model_name}")


def _run_detection(model, probe: Probe, img_w: int, img_h: int) -> dict:
    """Run detection model on a single probe, return prediction dict."""
    img_path = _get_image_path(probe)
    img_path_str = str(img_path) if img_path else "missing"

    t0 = time.time()

    from src.mock_model import MockCheatModel
    if isinstance(model, MockCheatModel):
        detections = model.predict(
            img_path_str, probe.prompt, img_w, img_h,
            target_box=probe.target_box,
            distractor_box=probe.distractor_box,
        )
    else:
        detections = model.predict(img_path_str, probe.prompt, img_w, img_h)

    wall_time = time.time() - t0

    candidates = []
    for det in detections:
        candidates.append({
            "box_xyxy": det.box_xyxy,
            "confidence": det.confidence,
            "mask_rle": det.mask_rle,
        })

    top_box = candidates[0]["box_xyxy"] if candidates else None
    top_mask = candidates[0]["mask_rle"] if candidates else None
    top_conf = candidates[0]["confidence"] if candidates else None

    return {
        "probe_id": probe.probe_id,
        "model": "detection",
        "candidates": candidates,
        "top_box": top_box,
        "top_confidence": top_conf,
        "top_mask": top_mask,
        "wall_time_s": round(wall_time, 4),
    }


def _run_oracle(sam, probe: Probe, img_w: int, img_h: int) -> dict:
    """Run oracle-box SAM on a single probe."""
    img_path = _get_image_path(probe)
    img_path_str = str(img_path) if img_path else "missing"

    t0 = time.time()
    mask_rle = sam.segment(img_path_str, probe.target_box, img_w, img_h)
    wall_time = time.time() - t0

    return {
        "probe_id": probe.probe_id,
        "model": "oracle_box",
        "candidates": [{
            "box_xyxy": list(probe.target_box),
            "confidence": 1.0,
            "mask_rle": mask_rle,
        }],
        "top_box": list(probe.target_box),
        "top_confidence": 1.0,
        "top_mask": mask_rle,
        "wall_time_s": round(wall_time, 4),
    }


def run_inference(
    model_name: str,
    probe_file: Path,
    out_dir: Path,
    oracle_box: bool = False,
    max_probes: int = 0,
) -> None:
    probes = load_probes(probe_file)
    pred_dir = out_dir / model_name
    if oracle_box:
        pred_dir = out_dir / f"{model_name}_oracle"
    pred_dir.mkdir(parents=True, exist_ok=True)

    if oracle_box:
        sam = _build_sam(model_name)
        probes = [p for p in probes if p.target_mask is not None]
        print(f"Oracle mode: {len(probes)} probes with masks")
    else:
        model = _build_model(model_name)

    if max_probes > 0:
        probes = probes[:max_probes]
        print(f"Smoke test: limited to {max_probes} probes")

    processed = 0
    skipped = 0
    empty = 0

    print(f"Running {model_name}{'_oracle' if oracle_box else ''} "
          f"on {len(probes)} probes → {pred_dir}/")

    for i, probe in enumerate(probes):
        out_path = pred_dir / f"{probe.probe_id}.json"
        if out_path.exists():
            skipped += 1
            continue

        img_w, img_h = _get_image_dims(probe)

        if oracle_box:
            result = _run_oracle(sam, probe, img_w, img_h)
        else:
            result = _run_detection(model, probe, img_w, img_h)

        if not result["candidates"]:
            empty += 1

        with out_path.open("w") as f:
            json.dump(result, f, indent=2)
            f.write("\n")

        processed += 1
        if (i + 1) % 100 == 0:
            print(f"  [{i + 1}/{len(probes)}] processed={processed} "
                  f"skipped={skipped} empty={empty}")

    print(f"\nDone: processed={processed}, skipped={skipped}, "
          f"detection-miss={empty}, total={len(probes)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True,
                        choices=["mock", "mock_cheat", "grounded_sam"])
    parser.add_argument("--probe-file", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("predictions"))
    parser.add_argument("--oracle-box", action="store_true",
                        help="Use ground-truth box + SAM instead of detection")
    parser.add_argument("--max-probes", type=int, default=0,
                        help="Limit to N probes (0=all, for smoke testing)")
    args = parser.parse_args()

    run_inference(args.model, args.probe_file, args.out_dir,
                  args.oracle_box, args.max_probes)


if __name__ == "__main__":
    main()

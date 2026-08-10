"""Grounded-SAM model wrapper.

GroundingDINO for open-vocabulary detection + SAM for segmentation.
Matches the same predict() / segment() interface as the mock models.

This module will only work on a machine with:
  - CUDA-capable GPU
  - groundingdino, segment_anything, torch installed
  - Downloaded weights (see setup_server.sh)

Usage from run_inference.py — not called directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.mock_model import Detection

# ── Default paths (override via constructor) ──────────────────────

WEIGHTS_DIR = Path("weights")

def _find_gdino_config() -> Path:
    """Locate GroundingDINO config file (works with pip install or git clone)."""
    candidates = [
        Path("groundingdino/config/GroundingDINO_SwinT_OGC.py"),
    ]
    try:
        import groundingdino
        pkg_dir = Path(groundingdino.__path__[0])
        candidates.insert(0, pkg_dir / "config" / "GroundingDINO_SwinT_OGC.py")
    except ImportError:
        pass
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


GDINO_CONFIG = _find_gdino_config()
GDINO_WEIGHTS = WEIGHTS_DIR / "groundingdino_swint_ogc.pth"

SAM_CHECKPOINT = WEIGHTS_DIR / "sam_vit_l_0b3195.pth"
SAM_MODEL_TYPE = "vit_l"


def _mask_to_rle(mask: np.ndarray) -> dict:
    """Convert a binary mask (H, W) to COCO compressed RLE.

    Uses pycocotools for speed. The mask must be uint8, Fortran-contiguous
    (column-major) for pycocotools.
    """
    import pycocotools.mask as mask_util

    h, w = mask.shape
    # pycocotools expects Fortran-order uint8 array
    fortran_mask = np.asfortranarray(mask.astype(np.uint8))
    rle = mask_util.encode(fortran_mask)
    # counts comes back as bytes — decode to str for JSON serialization
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def _xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """Convert xyxy to center-x, center-y, width, height (GroundingDINO format)."""
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return torch.stack([cx, cy, w, h], dim=-1)


def _cxcywh_to_xyxy(boxes: torch.Tensor, img_w: int, img_h: int) -> torch.Tensor:
    """Convert normalized cxcywh back to pixel xyxy."""
    cx, cy, w, h = boxes.unbind(-1)
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return torch.stack([x1, y1, x2, y2], dim=-1)


class GroundedSAMModel:
    """GroundingDINO detection + SAM segmentation.

    Parameters
    ----------
    gdino_config : Path
        Path to GroundingDINO config file.
    gdino_weights : Path
        Path to GroundingDINO checkpoint.
    sam_checkpoint : Path
        Path to SAM checkpoint.
    sam_model_type : str
        SAM model type (vit_l, vit_h, vit_b).
    device : str
        CUDA device.
    box_threshold : float
        GroundingDINO box confidence threshold.
    text_threshold : float
        GroundingDINO text confidence threshold.
    """

    def __init__(
        self,
        gdino_config: Path = GDINO_CONFIG,
        gdino_weights: Path = GDINO_WEIGHTS,
        sam_checkpoint: Path = SAM_CHECKPOINT,
        sam_model_type: str = SAM_MODEL_TYPE,
        device: str = "cuda",
        box_threshold: float = 0.25,
        text_threshold: float = 0.20,
    ):
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        # ── Load GroundingDINO ────────────────────────────────────
        from groundingdino.util.inference import load_model as load_gdino

        self.gdino = load_gdino(
            str(gdino_config), str(gdino_weights), device=device,
        )

        # ── Load SAM ─────────────────────────────────────────────
        from segment_anything import sam_model_registry, SamPredictor

        sam = sam_model_registry[sam_model_type](checkpoint=str(sam_checkpoint))
        sam.to(device)
        self.sam_predictor = SamPredictor(sam)

        self._last_image_path: str | None = None

    def _load_image_gdino(self, image_path: str):
        """Load image in GroundingDINO's expected format."""
        from groundingdino.util.inference import load_image

        # load_image returns (image_source_bgr, image_tensor)
        image_source, image_tensor = load_image(image_path)
        return image_source, image_tensor

    def _set_sam_image(self, image_path: str) -> np.ndarray:
        """Set SAM predictor's image (skip if already loaded)."""
        if self._last_image_path == image_path:
            return self._last_image_array

        import cv2
        image_bgr = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self.sam_predictor.set_image(image_rgb)
        self._last_image_path = image_path
        self._last_image_array = image_rgb
        return image_rgb

    def predict(
        self,
        image_path: str,
        prompt: str,
        img_w: int = 0,
        img_h: int = 0,
        **kwargs,
    ) -> list[Detection]:
        """Detect objects matching prompt, segment each with SAM.

        Parameters
        ----------
        image_path : str
            Path to the image file.
        prompt : str
            Text prompt (e.g., "the red car").
        img_w, img_h : int
            Image dimensions (used for fallback; actual dims read from image).

        Returns
        -------
        list[Detection]
            Detections sorted by confidence (highest first).
        """
        from groundingdino.util.inference import predict as gdino_predict

        # ── GroundingDINO detection ───────────────────────────────
        image_source, image_tensor = self._load_image_gdino(image_path)
        h, w = image_source.shape[:2]

        # GroundingDINO wants the prompt lowercased with a trailing period
        caption = prompt.lower().strip()
        if not caption.endswith("."):
            caption += "."

        boxes, logits, phrases = gdino_predict(
            model=self.gdino,
            image=image_tensor,
            caption=caption,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            device=self.device,
        )
        # boxes: tensor of shape (N, 4) in normalized cxcywh format
        # logits: tensor of shape (N,) — confidence scores

        if len(boxes) == 0:
            return []

        # Convert normalized cxcywh → pixel xyxy
        boxes_xyxy = _cxcywh_to_xyxy(boxes, w, h)
        # Clamp to image bounds
        boxes_xyxy[:, 0].clamp_(0, w)
        boxes_xyxy[:, 1].clamp_(0, h)
        boxes_xyxy[:, 2].clamp_(0, w)
        boxes_xyxy[:, 3].clamp_(0, h)

        # ── SAM segmentation ─────────────────────────────────────
        self._set_sam_image(image_path)

        detections: list[Detection] = []

        # SAM can process multiple boxes at once via transform
        boxes_np = boxes_xyxy.cpu().numpy()
        input_boxes = self.sam_predictor.transform.apply_boxes_torch(
            boxes_xyxy, (h, w),
        ).to(self.device)

        masks_batch, scores_batch, _ = self.sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=input_boxes,
            multimask_output=False,
        )
        # masks_batch: (N, 1, H, W), scores_batch: (N, 1)

        for i in range(len(boxes_np)):
            box = boxes_np[i].tolist()
            confidence = float(logits[i].item())

            # Extract mask: (H, W) binary
            mask = masks_batch[i, 0].cpu().numpy().astype(np.uint8)
            mask_rle = _mask_to_rle(mask)

            detections.append(Detection(
                box_xyxy=box,
                confidence=confidence,
                mask_rle=mask_rle,
            ))

        # Sort by confidence descending
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections


class GroundedSAM:
    """SAM-only wrapper for oracle-box segmentation.

    Given a ground-truth box, runs SAM to produce a segmentation mask.
    Used for the oracle-box evaluation path.
    """

    def __init__(
        self,
        sam_checkpoint: Path = SAM_CHECKPOINT,
        sam_model_type: str = SAM_MODEL_TYPE,
        device: str = "cuda",
    ):
        from segment_anything import sam_model_registry, SamPredictor

        self.device = device
        sam = sam_model_registry[sam_model_type](checkpoint=str(sam_checkpoint))
        sam.to(device)
        self.sam_predictor = SamPredictor(sam)
        self._last_image_path: str | None = None

    def _set_image(self, image_path: str) -> np.ndarray:
        """Load image into SAM predictor (cached)."""
        if self._last_image_path == image_path:
            return self._last_image_array

        import cv2
        image_bgr = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self.sam_predictor.set_image(image_rgb)
        self._last_image_path = image_path
        self._last_image_array = image_rgb
        return image_rgb

    def segment(
        self,
        image_path: str,
        box: tuple[float, ...],
        img_w: int = 0,
        img_h: int = 0,
    ) -> dict:
        """Segment using a ground-truth box prompt.

        Parameters
        ----------
        image_path : str
            Path to image.
        box : tuple
            Ground-truth box as (x1, y1, x2, y2) pixel coordinates.
        img_w, img_h : int
            Not used (dimensions read from image).

        Returns
        -------
        dict
            COCO compressed RLE mask.
        """
        image_rgb = self._set_image(image_path)

        box_array = np.array(box, dtype=np.float32).reshape(1, 4)
        box_torch = torch.from_numpy(box_array).to(self.device)

        masks, scores, _ = self.sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=box_torch,
            multimask_output=False,
        )
        # masks: (1, 1, H, W)
        mask = masks[0, 0].cpu().numpy().astype(np.uint8)
        return _mask_to_rle(mask)

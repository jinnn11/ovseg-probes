"""Mock models for pipeline testing.

MockModel: returns 0–3 random boxes with blob masks.
MockCheatModel: returns ground-truth box ~90% of the time.
MockSAM: returns a blob mask for a given box (oracle-box path).
"""

from __future__ import annotations

import random
import struct
from dataclasses import dataclass, field
from typing import Any

from src.schema import Box, RLE


@dataclass
class Detection:
    box_xyxy: list[float]
    confidence: float
    mask_rle: RLE


def _blob_rle(box: tuple[float, ...], img_h: int, img_w: int) -> RLE:
    """Create a filled-rectangle RLE mask for the given box."""
    x1 = max(0, int(round(box[0])))
    y1 = max(0, int(round(box[1])))
    x2 = min(img_w, int(round(box[2])))
    y2 = min(img_h, int(round(box[3])))

    # COCO RLE is column-major: iterate columns, encode runs
    counts: list[int] = []
    prev = 0
    run = 0
    for col in range(img_w):
        for row in range(img_h):
            val = 1 if (x1 <= col < x2 and y1 <= row < y2) else 0
            if val == prev:
                run += 1
            else:
                counts.append(run)
                run = 1
                prev = val
    counts.append(run)

    # Encode counts as COCO-style compressed string
    rle_str = _counts_to_rle_string(counts)
    return {"counts": rle_str, "size": [img_h, img_w]}


def _counts_to_rle_string(counts: list[int]) -> str:
    """Encode run-length counts as COCO compressed RLE string."""
    chars: list[str] = []
    for cnt in counts:
        more = True
        while more:
            c = cnt & 0x1F
            cnt >>= 5
            more = cnt != 0
            if more:
                c |= 0x20
            chars.append(chr(48 + c))
    return "".join(chars)


def _random_box(img_w: int, img_h: int, rng: random.Random) -> list[float]:
    """Generate a random box within image bounds."""
    min_side = 20
    w = rng.randint(min_side, max(min_side, img_w // 3))
    h = rng.randint(min_side, max(min_side, img_h // 3))
    x1 = rng.randint(0, max(0, img_w - w))
    y1 = rng.randint(0, max(0, img_h - h))
    return [float(x1), float(y1), float(x1 + w), float(y1 + h)]


class MockModel:
    """Returns 0–3 random detections per probe."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def predict(
        self, image_path: str, prompt: str, img_w: int = 640, img_h: int = 480,
    ) -> list[Detection]:
        n = self.rng.choices([0, 1, 2, 3], weights=[0.05, 0.60, 0.25, 0.10])[0]
        detections = []
        for _ in range(n):
            box = _random_box(img_w, img_h, self.rng)
            conf = round(self.rng.uniform(0.1, 0.95), 4)
            mask = _blob_rle(box, img_h, img_w)
            detections.append(Detection(box_xyxy=box, confidence=conf, mask_rle=mask))
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections


class MockCheatModel:
    """Returns ground-truth box ~90% of the time, distractor otherwise.

    Requires probe metadata (target_box, distractor_box) to be passed
    alongside the normal predict arguments.
    """

    def __init__(self, seed: int = 42, hit_rate: float = 0.90):
        self.rng = random.Random(seed)
        self.hit_rate = hit_rate

    def predict(
        self,
        image_path: str,
        prompt: str,
        img_w: int = 640,
        img_h: int = 480,
        *,
        target_box: tuple[float, ...] | None = None,
        distractor_box: tuple[float, ...] | None = None,
    ) -> list[Detection]:
        if target_box is None:
            return []

        if self.rng.random() < self.hit_rate:
            box = list(target_box)
            conf = round(self.rng.uniform(0.7, 0.99), 4)
        else:
            if distractor_box is not None:
                box = list(distractor_box)
            else:
                box = _random_box(img_w, img_h, self.rng)
            conf = round(self.rng.uniform(0.5, 0.85), 4)

        mask = _blob_rle(box, img_h, img_w)
        return [Detection(box_xyxy=box, confidence=conf, mask_rle=mask)]


class MockSAM:
    """Mock SAM: returns a blob mask for the given box (oracle-box path)."""

    def segment(
        self, image_path: str, box: tuple[float, ...],
        img_w: int = 640, img_h: int = 480,
    ) -> RLE:
        return _blob_rle(box, img_h, img_w)

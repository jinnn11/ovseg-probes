"""Geometry helpers.

All boxes accepted and returned by this module are xyxy pixel coordinates
unless a function name explicitly states another format.
"""

from __future__ import annotations

from typing import Sequence

Box = tuple[float, float, float, float]


def xywh_to_xyxy(box: Sequence[float]) -> Box:
    """Convert [x, y, width, height] to [x_min, y_min, x_max, y_max]."""
    if len(box) != 4:
        raise ValueError("xywh box must contain exactly 4 values")
    x, y, w, h = (float(value) for value in box)
    if w < 0 or h < 0:
        raise ValueError("xywh width and height must be non-negative")
    return (x, y, x + w, y + h)


def box_area(box: Sequence[float]) -> float:
    x1, y1, x2, y2 = _as_xyxy(box)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Return intersection-over-union for two xyxy boxes."""
    ax1, ay1, ax2, ay2 = _as_xyxy(box_a)
    bx1, by1, bx2, by2 = _as_xyxy(box_b)

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = box_area((ax1, ay1, ax2, ay2)) + box_area((bx1, by1, bx2, by2)) - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def box_center(box: Sequence[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = _as_xyxy(box)
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def is_left_of(box_a: Sequence[float], box_b: Sequence[float]) -> bool:
    return box_center(box_a)[0] < box_center(box_b)[0]


def is_right_of(box_a: Sequence[float], box_b: Sequence[float]) -> bool:
    return box_center(box_a)[0] > box_center(box_b)[0]


def _as_xyxy(box: Sequence[float]) -> Box:
    if len(box) != 4:
        raise ValueError("xyxy box must contain exactly 4 values")
    x1, y1, x2, y2 = (float(value) for value in box)
    if x2 < x1 or y2 < y1:
        raise ValueError("xyxy box must satisfy x_max >= x_min and y_max >= y_min")
    return (x1, y1, x2, y2)

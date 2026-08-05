"""Probe schema and JSON serialization helpers.

Project files store all boxes as xyxy pixel coordinates. Dataset-specific
coordinate formats must be converted before constructing Probe objects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable

Box = tuple[float, float, float, float]
RLE = dict[str, Any]


@dataclass(frozen=True)
class Probe:
    probe_id: str
    image_id: str | int
    image_source: str
    phenomenon: str
    prompt: str
    target_box: Box
    target_mask: RLE | None
    distractor_box: Box | None
    has_distractor: bool
    pair_id: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_box", _validate_box(self.target_box, "target_box"))
        if self.distractor_box is None:
            object.__setattr__(self, "distractor_box", None)
        else:
            object.__setattr__(
                self,
                "distractor_box",
                _validate_box(self.distractor_box, "distractor_box"),
            )

        for field_name in ("probe_id", "image_source", "phenomenon", "prompt", "notes"):
            value = getattr(self, field_name)
            if not isinstance(value, str):
                raise ValueError(f"{field_name} must be a string")

        if not self.probe_id:
            raise ValueError("probe_id must be non-empty")
        if not self.image_source:
            raise ValueError("image_source must be non-empty")
        if not self.phenomenon:
            raise ValueError("phenomenon must be non-empty")
        if not self.prompt:
            raise ValueError("prompt must be non-empty")
        if not isinstance(self.has_distractor, bool):
            raise ValueError("has_distractor must be a bool")
        if self.has_distractor != (self.distractor_box is not None):
            raise ValueError("has_distractor must match whether distractor_box is present")
        if self.target_mask is not None:
            _validate_rle(self.target_mask)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Probe":
        required = {
            "probe_id",
            "image_id",
            "image_source",
            "phenomenon",
            "prompt",
            "target_box",
            "target_mask",
            "distractor_box",
            "has_distractor",
            "notes",
        }
        optional = {"pair_id"}
        missing = required - set(raw)
        if missing:
            raise ValueError(f"Probe is missing required fields: {sorted(missing)}")
        extra = set(raw) - required - optional
        if extra:
            raise ValueError(f"Probe has unknown fields: {sorted(extra)}")
        return cls(**{k: v for k, v in raw.items() if k in required | optional})

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["target_box"] = list(self.target_box)
        if self.distractor_box is not None:
            data["distractor_box"] = list(self.distractor_box)
        return data


def load_probe(path: str | Path) -> Probe:
    data = _read_json(path)
    if not isinstance(data, dict):
        raise ValueError("Expected a single probe object")
    return Probe.from_dict(data)


def load_probes(path: str | Path) -> list[Probe]:
    data = _read_json(path)
    if isinstance(data, dict):
        return [Probe.from_dict(data)]
    if not isinstance(data, list):
        raise ValueError("Expected a probe object or list of probe objects")
    return [Probe.from_dict(item) for item in data]


def save_probe(probe: Probe, path: str | Path) -> None:
    _write_json(probe.to_dict(), path)


def save_probes(probes: Iterable[Probe], path: str | Path) -> None:
    _write_json([probe.to_dict() for probe in probes], path)


def _read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(data: Any, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def _validate_box(value: Any, field_name: str) -> Box:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{field_name} must be a 4-item xyxy box")

    try:
        x1, y1, x2, y2 = (float(coord) for coord in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} coordinates must be numeric") from exc

    if x2 < x1 or y2 < y1:
        raise ValueError(f"{field_name} must satisfy x_max >= x_min and y_max >= y_min")
    return (x1, y1, x2, y2)


def _validate_rle(value: Any) -> None:
    if not isinstance(value, dict):
        raise ValueError("target_mask must be an RLE dict or null")
    if "counts" not in value or "size" not in value:
        raise ValueError("target_mask RLE must include counts and size")
    size = value["size"]
    if not isinstance(size, list) or len(size) != 2 or not all(isinstance(n, int) for n in size):
        raise ValueError("target_mask RLE size must be [height, width]")

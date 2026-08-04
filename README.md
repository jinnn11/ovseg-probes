# ovseg-probes

Probe mining and evaluation utilities for open-vocabulary segmentation behavior.

## Coordinate Convention

All boxes committed by this project are `xyxy` pixel coordinates:

```text
[x_min, y_min, x_max, y_max]
```

COCO annotations use `xywh`:

```text
[x_min, y_min, width, height]
```

Convert COCO `xywh` boxes to project `xyxy` boxes exactly once at load time, before writing probes, predictions, or intermediate project JSON. Do not mix coordinate systems inside project files. Coordinate confusion is the easiest silent bug in this repo.

## Layout

- `data/`: local annotations and images, gitignored
- `probes/`: frozen probe JSONs, committed
- `predictions/`: local model outputs, gitignored
- `src/`: schema, geometry, mining, inference, and analysis code
- `notebooks/`: manual verification notebooks
- `tests/`: pytest tests

## Setup

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt
```

Run tests:

```bash
./.venv/bin/python -m pytest
```

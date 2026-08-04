"""Run segmentation model inference for frozen probes."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.schema import load_probes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probes_json", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("predictions"))
    args = parser.parse_args()

    probes = load_probes(args.probes_json)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raise NotImplementedError(f"Inference is not implemented yet for {len(probes)} probes.")


if __name__ == "__main__":
    main()

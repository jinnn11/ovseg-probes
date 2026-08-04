"""Analyze model predictions against frozen probes."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.schema import load_probes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probes_json", type=Path)
    parser.add_argument("predictions_dir", type=Path)
    args = parser.parse_args()

    probes = load_probes(args.probes_json)
    if not args.predictions_dir.exists():
        raise FileNotFoundError(args.predictions_dir)
    raise NotImplementedError(f"Analysis is not implemented yet for {len(probes)} probes.")


if __name__ == "__main__":
    main()

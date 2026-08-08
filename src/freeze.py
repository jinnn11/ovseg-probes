"""Compile verified probes into frozen probe_set_v1.json + control_set_v1.json.

Reads decisions.jsonl (keep/reject/flag) and all probe files.  Only probes
with decision="keep" are included.  Controls are kept if decided "keep" or
if undecided (controls from spot-checked categories may not all be reviewed).

Usage:
    python -m src.freeze
    python -m src.freeze --require-decision   # only include decided controls
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from src.schema import Probe, load_probes, save_probes

PROBES_DIR = Path("probes")
DECISIONS_FILE = Path("decisions.jsonl")
OUTPUT_DIR = Path("probes")

CONTROL_PHENOMENA = {
    "spatial_left_control", "spatial_right_control",
    "attribute_color_control",
    "finegrained_confusable_control", "finegrained_distinct_control",
    "negation_control",
}

DISTRACTOR_FILES = [
    "spatial_distractor.json",
    "finegrained_distractor.json",
    "attribute_distractor.json",
    "negation_distractor.json",
]

CONTROL_FILES = [
    "spatial_control.json",
    "finegrained_control.json",
    "attribute_control.json",
    "negation_control.json",
]


def freeze(require_decision: bool = False) -> None:
    # Load decisions
    decisions: dict[str, dict] = {}
    if DECISIONS_FILE.exists():
        with DECISIONS_FILE.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    d = json.loads(line)
                    decisions[d["probe_id"]] = d

    print(f"Loaded {len(decisions)} decisions from {DECISIONS_FILE}")

    # Load and filter distractor probes
    kept_probes: list[Probe] = []
    rejected = 0
    undecided_dist = 0
    for fname in DISTRACTOR_FILES:
        path = PROBES_DIR / fname
        if not path.exists():
            print(f"  Warning: {path} not found, skipping")
            continue
        probes = load_probes(path)
        for p in probes:
            d = decisions.get(p.probe_id)
            if d is None:
                undecided_dist += 1
            elif d["decision"] == "keep":
                kept_probes.append(p)
            else:
                rejected += 1

    # Load and filter control probes
    kept_controls: list[Probe] = []
    undecided_ctrl = 0
    for fname in CONTROL_FILES:
        path = PROBES_DIR / fname
        if not path.exists():
            continue
        probes = load_probes(path)
        for p in probes:
            d = decisions.get(p.probe_id)
            if d is None:
                if require_decision:
                    undecided_ctrl += 1
                else:
                    kept_controls.append(p)
                    undecided_ctrl += 1
            elif d["decision"] == "keep":
                kept_controls.append(p)
            else:
                rejected += 1

    # Save
    probe_out = OUTPUT_DIR / "probe_set_v1.json"
    control_out = OUTPUT_DIR / "control_set_v1.json"

    save_probes(kept_probes, probe_out)
    save_probes(kept_controls, control_out)

    # Summary
    print(f"\nFrozen probe set:")
    print(f"  {probe_out}: {len(kept_probes)} probes")
    print(f"  {control_out}: {len(kept_controls)} controls")
    print(f"  Rejected: {rejected}")
    print(f"  Undecided distractor: {undecided_dist}")
    if not require_decision:
        print(f"  Undecided controls included: {undecided_ctrl}")

    # Per-phenomenon breakdown
    phenom_counts: dict[str, int] = Counter()
    for p in kept_probes:
        phenom_counts[p.phenomenon] += 1
    ctrl_counts: dict[str, int] = Counter()
    for p in kept_controls:
        ctrl_counts[p.phenomenon] += 1

    print(f"\nPer-phenomenon (distractor):")
    for phenom, count in sorted(phenom_counts.items()):
        print(f"  {phenom:<35} {count}")
    print(f"\nPer-phenomenon (control):")
    for phenom, count in sorted(ctrl_counts.items()):
        print(f"  {phenom:<35} {count}")

    # Pair integrity check
    pair_ids = defaultdict(int)
    for p in kept_probes:
        if p.pair_id:
            pair_ids[p.pair_id] += 1
    complete = sum(1 for c in pair_ids.values() if c == 2)
    singletons = sum(1 for c in pair_ids.values() if c == 1)
    print(f"\nPair integrity:")
    print(f"  Complete pairs: {complete}")
    print(f"  Singletons (mirror rejected): {singletons}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-decision", action="store_true",
                        help="Only include controls with explicit keep decision")
    args = parser.parse_args()
    freeze(require_decision=args.require_decision)


if __name__ == "__main__":
    main()

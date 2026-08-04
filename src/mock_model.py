"""Mock model utilities for early pipeline tests."""

from __future__ import annotations

from dataclasses import dataclass

from src.schema import Probe


@dataclass(frozen=True)
class MockPrediction:
    probe_id: str
    box: tuple[float, float, float, float]
    score: float


class MockModel:
    """Returns the target box as a deterministic stand-in prediction."""

    def predict(self, probe: Probe) -> MockPrediction:
        return MockPrediction(probe_id=probe.probe_id, box=probe.target_box, score=1.0)

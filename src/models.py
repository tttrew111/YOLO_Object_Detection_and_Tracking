from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PersonTrack:
    box: tuple[int, int, int, int]
    track_id: int
    confidence: float
    raw_track_id: int | None = None

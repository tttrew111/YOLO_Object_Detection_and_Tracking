from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.models import PersonTrack


@dataclass
class TrackMemory:
    stable_id: int
    raw_track_id: int
    box: tuple[int, int, int, int]
    center: tuple[float, float]
    histogram: np.ndarray
    last_seen_frame: int


class StableIdAssigner:
    """Keeps display IDs stable when the underlying tracker changes IDs."""

    def __init__(self, max_missing_frames: int, match_threshold: float) -> None:
        self.max_missing_frames = max_missing_frames
        self.match_threshold = match_threshold
        self.next_stable_id = 1
        self.memories: dict[int, TrackMemory] = {}
        self.raw_to_stable: dict[int, int] = {}

    def assign(self, frame, tracks: list[PersonTrack], frame_index: int) -> list[PersonTrack]:
        self._drop_expired(frame_index)

        assigned_stable_ids: set[int] = set()
        stable_tracks: list[PersonTrack] = []

        for track in tracks:
            raw_id = track.track_id
            histogram = crop_histogram(frame, track.box)
            stable_id = self._match_stable_id(track, raw_id, histogram, frame, frame_index, assigned_stable_ids)

            assigned_stable_ids.add(stable_id)
            self.raw_to_stable[raw_id] = stable_id
            self.memories[stable_id] = TrackMemory(
                stable_id=stable_id,
                raw_track_id=raw_id,
                box=track.box,
                center=box_center(track.box),
                histogram=histogram,
                last_seen_frame=frame_index,
            )
            stable_tracks.append(
                PersonTrack(
                    box=track.box,
                    track_id=stable_id,
                    confidence=track.confidence,
                    raw_track_id=raw_id,
                )
            )

        return stable_tracks

    def _match_stable_id(
        self,
        track: PersonTrack,
        raw_id: int,
        histogram: np.ndarray,
        frame,
        frame_index: int,
        assigned_stable_ids: set[int],
    ) -> int:
        mapped_id = self.raw_to_stable.get(raw_id)
        if mapped_id in self.memories and mapped_id not in assigned_stable_ids:
            return mapped_id

        best_id = None
        best_score = self.match_threshold

        for stable_id, memory in self.memories.items():
            if stable_id in assigned_stable_ids:
                continue

            missing_frames = frame_index - memory.last_seen_frame
            score = match_score(track.box, histogram, memory, frame.shape[1], frame.shape[0], missing_frames)
            if score > best_score:
                best_id = stable_id
                best_score = score

        if best_id is not None:
            return best_id

        stable_id = self.next_stable_id
        self.next_stable_id += 1
        return stable_id

    def _drop_expired(self, frame_index: int) -> None:
        expired_ids = [
            stable_id
            for stable_id, memory in self.memories.items()
            if frame_index - memory.last_seen_frame > self.max_missing_frames
        ]

        for stable_id in expired_ids:
            del self.memories[stable_id]

        valid_ids = set(self.memories)
        self.raw_to_stable = {
            raw_id: stable_id
            for raw_id, stable_id in self.raw_to_stable.items()
            if stable_id in valid_ids
        }


def match_score(
    box: tuple[int, int, int, int],
    histogram: np.ndarray,
    memory: TrackMemory,
    frame_width: int,
    frame_height: int,
    missing_frames: int,
) -> float:
    appearance = histogram_similarity(histogram, memory.histogram)
    distance = center_distance(box_center(box), memory.center)
    gate = distance_gate(box, frame_width, frame_height, missing_frames)
    distance_score = max(0.0, 1.0 - (distance / gate))
    size_score = box_size_similarity(box, memory.box)

    return (0.55 * appearance) + (0.35 * distance_score) + (0.10 * size_score)


def crop_histogram(frame, box: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((256, 1), dtype=np.float32)

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist.astype(np.float32)


def histogram_similarity(left: np.ndarray, right: np.ndarray) -> float:
    similarity = cv2.compareHist(left, right, cv2.HISTCMP_CORREL)
    return float(max(0.0, min(1.0, similarity)))


def box_center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def center_distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return float(np.hypot(left[0] - right[0], left[1] - right[1]))


def distance_gate(
    box: tuple[int, int, int, int],
    frame_width: int,
    frame_height: int,
    missing_frames: int,
) -> float:
    x1, y1, x2, y2 = box
    box_scale = max(x2 - x1, y2 - y1, 1)
    frame_scale = np.hypot(frame_width, frame_height)
    missing_bonus = min(2.0, 1.0 + missing_frames / 60.0)
    return float(max(box_scale * 5.0, frame_scale * 0.12) * missing_bonus)


def box_size_similarity(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> float:
    left_area = box_area(left)
    right_area = box_area(right)
    if left_area <= 0 or right_area <= 0:
        return 0.0

    return min(left_area, right_area) / max(left_area, right_area)


def box_area(box: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = box
    return float(max(0, x2 - x1) * max(0, y2 - y1))

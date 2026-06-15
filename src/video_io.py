from __future__ import annotations

from pathlib import Path

import cv2


def open_video_capture(video_path: Path):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    return capture


def open_video_writer(output_path: str, fps: float, width: int, height: int):
    if not output_path:
        return None

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(output), fourcc, fps, (width, height))

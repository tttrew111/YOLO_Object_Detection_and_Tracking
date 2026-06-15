from __future__ import annotations

from pathlib import Path


PERSON_CLASS_ID = 0
DEFAULT_VIDEO_PATH = Path("video_input") / "input2.mp4"

# yolo11n is fast, but aerial/top-down people are small and harder to detect.
# yolo11s is a practical default for better recall without becoming too heavy.
DEFAULT_MODEL_PATH = "yolo11n.pt"
DEFAULT_CONFIDENCE = 0.20
DEFAULT_IOU = 0.50
DEFAULT_IMAGE_SIZE = 1280
DEFAULT_TRACKER = "botsort.yaml"
WINDOW_NAME = "YOLO11 Person ID Tracking"

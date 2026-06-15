from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from src.config import (
    DEFAULT_CONFIDENCE,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_IOU,
    DEFAULT_MODEL_PATH,
    DEFAULT_TRACKER,
    DEFAULT_VIDEO_PATH,
    WINDOW_NAME,
)
from src.detection import PersonDetector
from src.reid import StableIdAssigner
from src.ui import TrackingUI
from src.video_io import open_video_capture, open_video_writer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect people, assign tracker IDs, and track the clicked person."
    )
    parser.add_argument("--video", default=str(DEFAULT_VIDEO_PATH), help="Input video path.")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="YOLO model path or name.")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONFIDENCE, help="Detection confidence.")
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU, help="NMS IoU threshold.")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMAGE_SIZE, help="YOLO inference image size.")
    parser.add_argument("--tracker", default=DEFAULT_TRACKER, help="botsort.yaml or bytetrack.yaml.")
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Upscale frame before inference. Try 1.3-1.7 for small distant people.",
    )
    parser.add_argument(
        "--enhance",
        action="store_true",
        help="Apply CLAHE contrast enhancement before inference.",
    )
    parser.add_argument("--output", default="", help="Optional output video path.")
    parser.add_argument("--no-display", action="store_true", help="Run without the OpenCV preview window.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after this many frames. 0 means full video.")
    parser.add_argument("--no-reid", action="store_true", help="Use raw tracker IDs without stable ID reassignment.")
    parser.add_argument(
        "--reid-missing",
        type=int,
        default=120,
        help="Frames to remember lost people for stable ID reassignment.",
    )
    parser.add_argument(
        "--reid-threshold",
        type=float,
        default=0.42,
        help="Stable ID rematch threshold. Lower keeps IDs more aggressively.",
    )
    parser.add_argument("--show-raw-id", action="store_true", help="Display tracker raw ID next to stable ID.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    video_path = Path(args.video)

    if not video_path.exists():
        print(f"Video not found: {video_path}")
        return 1

    detector = PersonDetector(
        model_path=args.model,
        confidence=args.conf,
        iou=args.iou,
        image_size=args.imgsz,
        tracker_config=args.tracker,
        scale=args.scale,
        enhance=args.enhance,
    )
    stable_ids = None
    if not args.no_reid:
        stable_ids = StableIdAssigner(
            max_missing_frames=args.reid_missing,
            match_threshold=args.reid_threshold,
        )

    ui = TrackingUI()
    ui.show_raw_id = args.show_raw_id
    capture = open_video_capture(video_path)

    fps = capture.get(cv2.CAP_PROP_FPS) or 30
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = open_video_writer(args.output, fps, width, height)

    if not args.no_display:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, ui.on_mouse)

    last_frame = None
    processed_frames = 0

    try:
        while True:
            if args.max_frames > 0 and processed_frames >= args.max_frames:
                break

            if not ui.paused:
                success, frame = capture.read()
                if not success:
                    break

                people = detector.track_people(frame)
                processed_frames += 1
                if stable_ids is not None:
                    people = stable_ids.assign(frame, people, processed_frames)
                ui.latest_people = people
                ui.draw(frame, people)
                last_frame = frame

                if writer is not None:
                    writer.write(frame)

            if last_frame is None:
                continue

            key = show_frame(last_frame, args.no_display)
            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                ui.clear_selection()
            if key == ord(" "):
                ui.paused = not ui.paused
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    return 0


def show_frame(frame, no_display: bool) -> int:
    if no_display:
        return 255

    cv2.imshow(WINDOW_NAME, frame)
    return cv2.waitKey(1) & 0xFF

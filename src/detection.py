from __future__ import annotations

import cv2
from ultralytics import YOLO

from src.config import PERSON_CLASS_ID
from src.models import PersonTrack


class PersonDetector:
    def __init__(
        self,
        model_path: str,
        confidence: float,
        iou: float,
        image_size: int,
        tracker_config: str,
        scale: float,
        enhance: bool,
    ) -> None:
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.iou = iou
        self.image_size = image_size
        self.tracker_config = tracker_config
        self.scale = max(1.0, scale)
        self.enhance = enhance

    def track_people(self, frame) -> list[PersonTrack]:
        inference_frame = self._prepare_frame(frame)

        results = self.model.track(
            inference_frame,
            persist=True,
            classes=[PERSON_CLASS_ID],
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.image_size,
            tracker=self.tracker_config,
            agnostic_nms=True,
            verbose=False,
        )

        return self._extract_people(results[0], frame.shape[1], frame.shape[0])

    def _prepare_frame(self, frame):
        prepared = frame

        if self.scale > 1.0:
            prepared = cv2.resize(
                prepared,
                None,
                fx=self.scale,
                fy=self.scale,
                interpolation=cv2.INTER_CUBIC,
            )

        if self.enhance:
            prepared = enhance_contrast(prepared)

        return prepared

    def _extract_people(self, result, frame_width: int, frame_height: int) -> list[PersonTrack]:
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        track_ids = boxes.id.cpu().numpy().astype(int)
        classes = boxes.cls.cpu().numpy().astype(int)
        confidences = boxes.conf.cpu().numpy()

        people: list[PersonTrack] = []
        for box, track_id, class_id, confidence in zip(xyxy, track_ids, classes, confidences):
            if class_id != PERSON_CLASS_ID:
                continue

            x1, y1, x2, y2 = [int(value / self.scale) for value in box]
            people.append(
                PersonTrack(
                    box=clip_box((x1, y1, x2, y2), frame_width, frame_height),
                    track_id=int(track_id),
                    confidence=float(confidence),
                )
            )

        return people


def enhance_contrast(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_lightness = clahe.apply(lightness)
    enhanced = cv2.merge((enhanced_lightness, channel_a, channel_b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def clip_box(
    box: tuple[int, int, int, int],
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (
        max(0, min(frame_width - 1, x1)),
        max(0, min(frame_height - 1, y1)),
        max(0, min(frame_width - 1, x2)),
        max(0, min(frame_height - 1, y2)),
    )

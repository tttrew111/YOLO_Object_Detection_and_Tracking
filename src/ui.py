from __future__ import annotations

from typing import Optional

import cv2

from src.models import PersonTrack


class TrackingUI:
    def __init__(self) -> None:
        self.selected_track_id: Optional[int] = None
        self.latest_people: list[PersonTrack] = []
        self.paused = False
        self.show_raw_id = False

    def on_mouse(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        for person in self.latest_people:
            x1, y1, x2, y2 = person.box
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.selected_track_id = person.track_id
                print(f"Selected ID: {self.selected_track_id}")
                return

        self.clear_selection()

    def clear_selection(self) -> None:
        self.selected_track_id = None
        print("Selection cleared")

    def draw(self, frame, people: list[PersonTrack]) -> None:
        selected_seen = False

        for person in people:
            x1, y1, x2, y2 = person.box
            is_selected = person.track_id == self.selected_track_id
            selected_seen = selected_seen or is_selected

            color = (0, 255, 255) if is_selected else (0, 180, 0)
            thickness = 4 if is_selected else 2
            label = self._label_for(person)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            draw_label(frame, label, x1, y1, color)

            if is_selected:
                draw_selected_marker(frame, person.box, color)

        self._draw_hud(frame, selected_seen)

    def _label_for(self, person: PersonTrack) -> str:
        label = f"ID {person.track_id} person {person.confidence:.2f}"
        if self.show_raw_id and person.raw_track_id is not None:
            label = f"{label} raw {person.raw_track_id}"
        return label

    def _draw_hud(self, frame, selected_seen: bool) -> None:
        status = "none" if self.selected_track_id is None else str(self.selected_track_id)
        lines = [
            f"Selected ID: {status}",
            "Click person | c: clear | space: pause | q/esc: quit",
        ]

        if self.paused:
            lines.append("PAUSED")
        if self.selected_track_id is not None and not selected_seen:
            lines.append(f"Selected ID {self.selected_track_id} temporarily lost")

        y = 28
        for line in lines:
            cv2.putText(
                frame,
                line,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            y += 28


def draw_selected_marker(frame, box: tuple[int, int, int, int], color: tuple[int, int, int]) -> None:
    x1, y1, x2, y2 = box
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    cv2.circle(frame, (center_x, center_y), 6, color, -1)
    cv2.putText(
        frame,
        "TRACKING",
        (x1, min(frame.shape[0] - 12, y2 + 28)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_label(frame, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
    top = max(0, y - height - baseline - 8)
    cv2.rectangle(frame, (x, top), (x + width + 8, top + height + baseline + 8), color, -1)
    cv2.putText(
        frame,
        text,
        (x + 4, top + height + 2),
        font,
        scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )

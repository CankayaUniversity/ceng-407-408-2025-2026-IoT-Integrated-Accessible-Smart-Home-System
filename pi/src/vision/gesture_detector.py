import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

from events.models import build_event
from vision.eye_tracker import (
    calc_ear,
    iris_relative_position,
    classify_gaze,
    RIGHT_EYE_IDX,
    LEFT_EYE_IDX,
    RIGHT_IRIS_CENTER,
    LEFT_IRIS_CENTER,
)


class GestureDetector:
    def __init__(self):
        self.short_blink_min = 0.08
        self.short_blink_max = 0.45
        self.long_blink_min = 0.45
        self.long_blink_max = 1.8

        self.ear_baseline = 0.28
        self.ear_baseline_alpha = 0.06
        self.min_drop_for_blink = 0.035

        self.close_frames_needed = 2
        self.open_frames_needed = 2

        self.close_streak = 0
        self.open_streak = 0
        self.blink_closed = False
        self.blink_start_time = 0.0
        self.blink_min_ear = 1.0

        self.gaze_stable_time = 0.35

        self.last_gaze = None
        self.gaze_same_time = 0.0
        self.gaze_event_fired = False

    def _blink_thresholds(self):
        close_th = float(max(0.14, min(0.30, self.ear_baseline * 0.68)))
        open_th = float(max(close_th + 0.015, min(0.34, self.ear_baseline * 0.82)))
        return close_th, open_th

    def process_face_landmarks(self, face_landmarks, frame_w: int, frame_h: int, now: float, dt: float):
        right_ear = calc_ear(face_landmarks, RIGHT_EYE_IDX, frame_w, frame_h)
        left_ear = calc_ear(face_landmarks, LEFT_EYE_IDX, frame_w, frame_h)
        ear = (right_ear + left_ear) / 2.0

        close_th, open_th = self._blink_thresholds()
        if (not self.blink_closed) and ear > close_th:
            self.ear_baseline = (1.0 - self.ear_baseline_alpha) * self.ear_baseline + self.ear_baseline_alpha * ear
            close_th, open_th = self._blink_thresholds()

        rx, ry = iris_relative_position(face_landmarks, RIGHT_IRIS_CENTER, RIGHT_EYE_IDX, frame_w, frame_h)
        lx, ly = iris_relative_position(face_landmarks, LEFT_IRIS_CENTER, LEFT_EYE_IDX, frame_w, frame_h)

        iris_x = (rx + lx) / 2.0
        iris_y = (ry + ly) / 2.0
        gaze_text = classify_gaze(iris_x, iris_y)

        if ear <= close_th:
            self.close_streak += 1
            self.open_streak = 0
        elif ear >= open_th:
            self.open_streak += 1
            self.close_streak = 0
        else:
            self.close_streak = 0
            self.open_streak = 0

        if (not self.blink_closed) and self.close_streak >= self.close_frames_needed:
            self.blink_closed = True
            self.blink_start_time = now
            self.blink_min_ear = ear

        if self.blink_closed:
            self.blink_min_ear = min(self.blink_min_ear, ear)

        if self.blink_closed and self.open_streak >= self.open_frames_needed:
            duration = now - self.blink_start_time
            blink_drop = self.ear_baseline - self.blink_min_ear

            self.blink_closed = False
            self.close_streak = 0
            self.open_streak = 0

            if blink_drop >= self.min_drop_for_blink:
                if self.short_blink_min <= duration <= self.short_blink_max:
                    return build_event(
                        "short_blink",
                        metadata={
                            "duration": duration,
                            "ear": ear,
                            "ear_min": self.blink_min_ear,
                            "ear_base": self.ear_baseline,
                        },
                        source="eye",
                    )
                if self.long_blink_min <= duration <= self.long_blink_max:
                    return build_event(
                        "long_blink",
                        metadata={
                            "duration": duration,
                            "ear": ear,
                            "ear_min": self.blink_min_ear,
                            "ear_base": self.ear_baseline,
                        },
                        source="eye",
                    )

        if gaze_text in ("LEFT", "RIGHT"):
            if gaze_text == self.last_gaze:
                self.gaze_same_time += dt
            else:
                self.last_gaze = gaze_text
                self.gaze_same_time = dt
                self.gaze_event_fired = False
        else:
            self.last_gaze = gaze_text
            self.gaze_same_time = 0.0
            self.gaze_event_fired = False

        if (
            gaze_text in ("LEFT", "RIGHT")
            and not self.gaze_event_fired
            and self.gaze_same_time >= self.gaze_stable_time
        ):
            self.gaze_event_fired = True
            event_name = "look_left" if gaze_text == "LEFT" else "look_right"
            return build_event(
                event_name,
                metadata={"gaze": gaze_text, "ear": ear},
                source="eye",
            )

        return None

    def reset(self):
        self.last_gaze = "NO_FACE"
        self.gaze_same_time = 0.0
        self.gaze_event_fired = False
        self.close_streak = 0
        self.open_streak = 0
        self.blink_closed = False
        self.blink_min_ear = 1.0


@dataclass
class HandState:
    center_x: float
    center_y: float
    index_x: float
    index_y: float
    pinch_ratio: float
    is_pinching: bool
    is_open_palm: bool
    is_pointing: bool
    timestamp: float


def _euclidean_distance(a, b) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def _fingers_extended(landmarks) -> dict:
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]

    index_tip = landmarks[8]
    index_pip = landmarks[6]
    middle_tip = landmarks[12]
    middle_pip = landmarks[10]
    ring_tip = landmarks[16]
    ring_pip = landmarks[14]
    pinky_tip = landmarks[20]
    pinky_pip = landmarks[18]

    return {
        "thumb": abs(thumb_tip.x - thumb_ip.x) > 0.035,
        "index": index_tip.y < index_pip.y,
        "middle": middle_tip.y < middle_pip.y,
        "ring": ring_tip.y < ring_pip.y,
        "pinky": pinky_tip.y < pinky_pip.y,
    }


class HandGestureDetector:
    def __init__(
        self,
        history_seconds: float = 0.45,
        swipe_min_dx: float = 0.22,
        swipe_max_dy: float = 0.16,
        pinch_threshold: float = 0.34,
        open_palm_hold_seconds: float = 0.8,
        cooldown_seconds: float = 0.7,
        release_seconds: float = 0.25,
        swipe_rearm_dx: float = 0.10,
        invert_swipe: bool = False,
    ) -> None:
        self.history_seconds = history_seconds
        self.swipe_min_dx = swipe_min_dx
        self.swipe_max_dy = swipe_max_dy
        self.pinch_threshold = pinch_threshold
        self.open_palm_hold_seconds = open_palm_hold_seconds
        self.cooldown_seconds = cooldown_seconds
        self.release_seconds = release_seconds
        self.swipe_rearm_dx = swipe_rearm_dx
        self.invert_swipe = invert_swipe

        self.history: Deque[HandState] = deque()
        self.last_event_name: Optional[str] = None
        self.last_event_time: float = 0.0
        self.open_palm_start_time: Optional[float] = None
        self.was_pinching: bool = False
        self.locked_event_name: Optional[str] = None
        self.release_start_time: Optional[float] = None
        self.locked_swipe_x: Optional[float] = None

    def process_hand_landmarks(self, hand_landmarks, now: Optional[float] = None):
        now = now or time.time()
        landmarks = hand_landmarks.landmark

        center_x = sum(lm.x for lm in landmarks) / len(landmarks)
        center_y = sum(lm.y for lm in landmarks) / len(landmarks)

        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        wrist = landmarks[0]
        middle_mcp = landmarks[9]

        pinch_distance = _euclidean_distance(thumb_tip, index_tip)
        hand_scale = max(_euclidean_distance(wrist, middle_mcp), 1e-6)
        pinch_ratio = pinch_distance / hand_scale
        is_pinching = pinch_ratio < self.pinch_threshold

        finger_state = _fingers_extended(landmarks)
        extended_count = sum(1 for value in finger_state.values() if value)
        is_open_palm = extended_count >= 4

        is_pointing = (
            finger_state["index"]
            and not finger_state["middle"]
            and not finger_state["ring"]
            and not finger_state["pinky"]
        )

        state = HandState(
            center_x=center_x,
            center_y=center_y,
            index_x=index_tip.x,
            index_y=index_tip.y,
            pinch_ratio=pinch_ratio,
            is_pinching=is_pinching,
            is_open_palm=is_open_palm,
            is_pointing=is_pointing,
            timestamp=now,
        )

        self.history.append(state)
        self._trim_history(now)

        event_name = self._detect_event(state)
        if event_name is None:
            return None

        if not self._can_emit(now):
            return None

        self.last_event_name = event_name
        self.last_event_time = now
        self._lock_until_released(event_name, state)

        return build_event(
            event_name,
            source="hand",
            metadata={
                "pinch_ratio": state.pinch_ratio,
                "is_pointing": state.is_pointing,
                "is_open_palm": state.is_open_palm,
                "index_x": state.index_x,
                "index_y": state.index_y,
            },
        )

    def reset(self) -> None:
        self.history.clear()
        self.open_palm_start_time = None
        self.was_pinching = False
        self.locked_event_name = None
        self.release_start_time = None
        self.locked_swipe_x = None

    def _trim_history(self, now: float) -> None:
        while self.history and now - self.history[0].timestamp > self.history_seconds:
            self.history.popleft()

    def _detect_event(self, state: HandState) -> Optional[str]:
        if self._is_locked(state):
            return None

        if state.is_pinching and not self.was_pinching:
            self.was_pinching = True
            self.open_palm_start_time = None
            return "pinch"

        if not state.is_pinching:
            self.was_pinching = False

        swipe_event = self._detect_pointing_swipe()
        if swipe_event:
            self.open_palm_start_time = None
            return swipe_event

        if state.is_open_palm and not state.is_pinching and not state.is_pointing:
            if self.open_palm_start_time is None:
                self.open_palm_start_time = state.timestamp
            elif state.timestamp - self.open_palm_start_time >= self.open_palm_hold_seconds:
                self.open_palm_start_time = state.timestamp
                return "open_palm_hold"
        else:
            self.open_palm_start_time = None

        return None

    def _detect_pointing_swipe(self) -> Optional[str]:
        if len(self.history) < 4:
            return None

        pointing_count = sum(1 for item in self.history if item.is_pointing and not item.is_pinching)
        if pointing_count < max(3, int(len(self.history) * 0.65)):
            return None

        oldest = self.history[0]
        newest = self.history[-1]
        dx = newest.index_x - oldest.index_x
        dy = newest.index_y - oldest.index_y

        if abs(dy) > self.swipe_max_dy:
            return None

        if dx <= -self.swipe_min_dx:
            self.history.clear()
            return self._map_swipe_event("swipe_left")

        if dx >= self.swipe_min_dx:
            self.history.clear()
            return self._map_swipe_event("swipe_right")

        return None

    def _map_swipe_event(self, event_name: str) -> str:
        if not self.invert_swipe:
            return event_name
        if event_name == "swipe_left":
            return "swipe_right"
        if event_name == "swipe_right":
            return "swipe_left"
        return event_name

    def _can_emit(self, now: float) -> bool:
        return now - self.last_event_time >= self.cooldown_seconds

    def _lock_until_released(self, event_name: str, state: HandState) -> None:
        self.locked_event_name = event_name
        self.release_start_time = None

        if event_name in {"swipe_left", "swipe_right"}:
            self.locked_swipe_x = state.index_x
        else:
            self.locked_swipe_x = None

    def _is_locked(self, state: HandState) -> bool:
        if self.locked_event_name is None:
            return False

        now = state.timestamp
        released = self._is_released_from_locked_event(state)

        if released:
            if self.release_start_time is None:
                self.release_start_time = now
            elif now - self.release_start_time >= self.release_seconds:
                self.locked_event_name = None
                self.release_start_time = None
                self.locked_swipe_x = None
                self.history.clear()
                return False
        else:
            self.release_start_time = None

        return True

    def _is_released_from_locked_event(self, state: HandState) -> bool:
        if self.locked_event_name == "pinch":
            return not state.is_pinching

        if self.locked_event_name == "open_palm_hold":
            return not state.is_open_palm

        if self.locked_event_name == "swipe_right":
            if self.locked_swipe_x is None:
                return not state.is_pointing
            return state.is_pointing and state.index_x <= self.locked_swipe_x - self.swipe_rearm_dx

        if self.locked_event_name == "swipe_left":
            if self.locked_swipe_x is None:
                return not state.is_pointing
            return state.is_pointing and state.index_x >= self.locked_swipe_x + self.swipe_rearm_dx

        return not state.is_pinching and not state.is_open_palm and not state.is_pointing
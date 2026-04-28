"""
Raspberry Pi Hand Gesture Detector

Purpose:
- Read frames from an IP camera / webcam
- Detect hand landmarks using MediaPipe Hands
- Classify UI-control gestures:
    hand:swipe_left
    hand:swipe_right
    hand:pinch
    hand:open_palm_hold
- Print events locally
- Optionally send events to backend POST /event

Recommended environment based on current project Pi setup:
    Python 3.11.x
    mediapipe==0.10.18
    opencv-python==4.11.0.86
    numpy==1.26.4
    requests

Example usage:
    # Uses CAMERA_URL from .env
    python pi_hand_gesture_detector.py --display --flip

    # CLI camera argument still overrides .env value when needed
    python pi_hand_gesture_detector.py --camera 0 --display --flip
    python pi_hand_gesture_detector.py --camera http://PHONE_IP:PORT/video --display --flip

Notes:
- This script does not directly control devices.
- It only generates standardized events.
- Backend mapping should decide what each event means.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Optional, Tuple

import cv2
import mediapipe as mp

try:
    import requests
except ModuleNotFoundError:
    requests = None


# -----------------------------
# Data models
# -----------------------------

@dataclass
class GestureEvent:
    event_type: str
    name: str
    source: str = "raspberry_pi"
    timestamp: float = 0.0

    def to_payload(self) -> dict:
        return {
            "event_type": self.event_type,
            "name": self.name,
            "source": self.source,
            "timestamp": self.timestamp or time.time(),
        }


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


# -----------------------------
# .env configuration
# -----------------------------

def find_env_file(start_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Search for a .env file starting from the current working directory and then
    walking up parent directories. This lets the script work when run from:
        pi/
        pi/src/
        pi/src/vision/
    """
    current = (start_dir or Path.cwd()).resolve()
    candidates = [current, *current.parents]

    for directory in candidates:
        env_path = directory / ".env"
        if env_path.exists():
            return env_path

    return None


def load_env_file() -> Optional[Path]:
    """
    Minimal .env loader to avoid requiring python-dotenv.
    Existing OS environment variables are not overwritten.
    """
    env_path = find_env_file()
    if not env_path:
        return None

    with env_path.open("r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            if key and key not in os.environ:
                os.environ[key] = value

    return env_path


def env_value(*names: str, default: Optional[str] = None) -> Optional[str]:
    for name in names:
        value = os.getenv(name)
        if value is not None and value != "":
            return value
    return default


def env_int(*names: str, default: int) -> int:
    value = env_value(*names)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def env_float(*names: str, default: float) -> float:
    value = env_value(*names)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def env_bool(*names: str, default: bool = False) -> bool:
    value = env_value(*names)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


ENV_FILE = load_env_file()


# -----------------------------
# Utility functions
# -----------------------------

def euclidean_distance(a, b) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def landmark_xy(landmark, width: int, height: int) -> Tuple[int, int]:
    return int(landmark.x * width), int(landmark.y * height)


def fingers_extended(landmarks) -> dict:
    """
    Simple finger-state estimator.

    For index/middle/ring/pinky:
    A finger is considered extended if its tip is above its pip joint in image coordinates.
    In OpenCV image coordinates, smaller y means higher position.

    Thumb detection is harder because it depends on hand orientation, so we use a basic
    horizontal distance heuristic.
    """
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


# -----------------------------
# Gesture classifier
# -----------------------------

class HandGestureClassifier:
    def __init__(
        self,
        history_seconds: float = 0.45,
        swipe_min_dx: float = 0.22,
        swipe_max_dy: float = 0.16,
        pinch_threshold: float = 0.34,
        open_palm_hold_seconds: float = 0.8,
        cooldown_seconds: float = 1.0,
        release_seconds: float = 0.35,
        swipe_rearm_dx: float = 0.10,
    ) -> None:
        self.history_seconds = history_seconds
        self.swipe_min_dx = swipe_min_dx
        self.swipe_max_dy = swipe_max_dy
        self.pinch_threshold = pinch_threshold
        self.open_palm_hold_seconds = open_palm_hold_seconds
        self.cooldown_seconds = cooldown_seconds
        self.release_seconds = release_seconds
        self.swipe_rearm_dx = swipe_rearm_dx

        self.history: Deque[HandState] = deque()
        self.last_event_name: Optional[str] = None
        self.last_event_time: float = 0.0
        self.open_palm_start_time: Optional[float] = None
        self.was_pinching: bool = False
        self.locked_event_name: Optional[str] = None
        self.release_start_time: Optional[float] = None
        self.locked_swipe_x: Optional[float] = None

    def update(self, landmarks) -> Tuple[Optional[GestureEvent], HandState]:
        now = time.time()

        center_x = sum(lm.x for lm in landmarks) / len(landmarks)
        center_y = sum(lm.y for lm in landmarks) / len(landmarks)

        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        wrist = landmarks[0]
        middle_mcp = landmarks[9]

        pinch_distance = euclidean_distance(thumb_tip, index_tip)
        hand_scale = max(euclidean_distance(wrist, middle_mcp), 1e-6)
        pinch_ratio = pinch_distance / hand_scale
        is_pinching = pinch_ratio < self.pinch_threshold

        finger_state = fingers_extended(landmarks)
        extended_count = sum(1 for value in finger_state.values() if value)
        is_open_palm = extended_count >= 4

        # Pointing pose: index finger is extended while middle/ring/pinky are folded.
        # Swipe detection is intentionally limited to this pose so open-palm movement
        # does not accidentally trigger swipe events.
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
            return None, state

        if not self._can_emit(event_name, now):
            return None, state

        self.last_event_name = event_name
        self.last_event_time = now
        self._lock_until_released(event_name, state)

        return GestureEvent(event_type="hand", name=event_name, timestamp=now), state

    def reset_when_no_hand(self) -> None:
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
        # If an event was just emitted, do not emit another one until the user
        # releases that gesture. This prevents stacked commands in UI navigation.
        if self._is_locked(state):
            return None

        # Pinch should behave like a click: emit only on transition non-pinch -> pinch.
        if state.is_pinching and not self.was_pinching:
            self.was_pinching = True
            self.open_palm_start_time = None
            return "pinch"

        if not state.is_pinching:
            self.was_pinching = False

        # Swipe detection now requires pointing pose and tracks index fingertip movement.
        # This avoids the conflict where moving an open palm triggers both open_palm_hold
        # and swipe events.
        swipe_event = self._detect_pointing_swipe()
        if swipe_event:
            self.open_palm_start_time = None
            return swipe_event

        # Open palm hold: emit after hand remains open for enough time.
        # Note: MediaPipe 2D landmarks cannot reliably distinguish palm side from back side.
        # We therefore keep open_palm_hold as a hold-only command, not a movement command.
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
        # Swipe detection needs enough movement history.
        if len(self.history) < 4:
            return None

        # Require most frames in the short history window to be pointing pose.
        pointing_count = sum(1 for item in self.history if item.is_pointing and not item.is_pinching)
        if pointing_count < max(3, int(len(self.history) * 0.65)):
            return None

        oldest = self.history[0]
        newest = self.history[-1]
        dx = newest.index_x - oldest.index_x
        dy = newest.index_y - oldest.index_y

        # Ignore diagonal / vertical movement for first prototype.
        if abs(dy) > self.swipe_max_dy:
            return None

        # Image x grows from left to right.
        if dx <= -self.swipe_min_dx:
            self.history.clear()
            return "swipe_left"

        if dx >= self.swipe_min_dx:
            self.history.clear()
            return "swipe_right"

        return None

    def _can_emit(self, event_name: str, now: float) -> bool:
        # Global cooldown: even different gestures cannot be emitted too closely.
        # This is important for UI navigation because one physical movement should
        # correspond to exactly one UI action.
        if now - self.last_event_time < self.cooldown_seconds:
            return False
        return True

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
            # After a right swipe, allow another right swipe only after the index
            # finger comes back toward the left/center by a small amount.
            if self.locked_swipe_x is None:
                return not state.is_pointing
            return state.is_pointing and state.index_x <= self.locked_swipe_x - self.swipe_rearm_dx

        if self.locked_event_name == "swipe_left":
            # After a left swipe, allow another left swipe only after the index
            # finger comes back toward the right/center by a small amount.
            if self.locked_swipe_x is None:
                return not state.is_pointing
            return state.is_pointing and state.index_x >= self.locked_swipe_x + self.swipe_rearm_dx

        return not state.is_pinching and not state.is_open_palm and not state.is_pointing


# -----------------------------
# Event sender
# -----------------------------

class EventSender:
    def __init__(self, backend_base_url: Optional[str], timeout_seconds: float = 1.5) -> None:
        self.backend_base_url = backend_base_url.rstrip("/") if backend_base_url else None
        self.timeout_seconds = timeout_seconds

    def emit(self, event: GestureEvent) -> None:
        payload = event.to_payload()
        print(f"EVENT -> {payload}")

        if not self.backend_base_url:
            return

        if requests is None:
            print("Backend sending is disabled because 'requests' is not installed.")
            return

        url = f"{self.backend_base_url}/event"
        try:
            response = requests.post(url, json=payload, timeout=self.timeout_seconds)
            if response.status_code >= 400:
                print(f"Backend error {response.status_code}: {response.text[:200]}")
        except requests.RequestException as exc:
            print(f"Could not send event to backend: {exc}")


# -----------------------------
# Main application
# -----------------------------

def parse_camera_source(value: str):
    if value.isdigit():
        return int(value)
    return value


def run(args) -> None:
    camera_source = parse_camera_source(args.camera)
    cap = cv2.VideoCapture(camera_source)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera source: {args.camera}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    classifier = HandGestureClassifier(
        history_seconds=args.history_seconds,
        swipe_min_dx=args.swipe_min_dx,
        swipe_max_dy=args.swipe_max_dy,
        pinch_threshold=args.pinch_threshold,
        open_palm_hold_seconds=args.open_palm_hold_seconds,
        cooldown_seconds=args.cooldown_seconds,
        release_seconds=args.release_seconds,
        swipe_rearm_dx=args.swipe_rearm_dx,
    )
    sender = EventSender(args.backend)

    last_fps_time = time.time()
    frame_count = 0
    fps = 0.0
    last_label = "no_hand"
    last_state: Optional[HandState] = None

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=args.model_complexity,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    ) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Could not read frame from camera. Retrying...")
                time.sleep(0.2)
                continue

            frame = cv2.resize(frame, (args.width, args.height))
            frame = cv2.flip(frame, 1) if args.flip else frame

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)
            rgb.flags.writeable = True

            frame_count += 1
            now = time.time()
            if now - last_fps_time >= 1.0:
                fps = frame_count / (now - last_fps_time)
                frame_count = 0
                last_fps_time = now

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                landmarks = hand_landmarks.landmark
                event, state = classifier.update(landmarks)
                last_state = state

                if event:
                    last_label = event.name
                    sender.emit(event)
                else:
                    if state.is_pinching:
                        last_label = "pinch_detected"
                    elif state.is_pointing:
                        last_label = "pointing_detected"
                    elif state.is_open_palm:
                        last_label = "open_palm_detected"
                    else:
                        last_label = "tracking_hand"

                if args.display:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style(),
                    )
            else:
                classifier.reset_when_no_hand()
                last_label = "no_hand"
                last_state = None

            if args.display:
                draw_overlay(frame, fps, last_label, last_state, args.backend)
                cv2.imshow("Pi Hand Gesture Detector", frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

    cap.release()
    cv2.destroyAllWindows()


def draw_overlay(frame, fps: float, label: str, state: Optional[HandState], backend: Optional[str]) -> None:
    y = 28
    line_height = 28

    cv2.putText(frame, f"FPS: {fps:.1f}", (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y += line_height

    cv2.putText(frame, f"State: {label}", (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y += line_height

    if state:
        cv2.putText(
            frame,
            f"Pinch ratio: {state.pinch_ratio:.2f}",
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        y += line_height
        cv2.putText(
            frame,
            f"Center: x={state.center_x:.2f}, y={state.center_y:.2f}",
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        y += line_height
        cv2.putText(
            frame,
            f"Index: x={state.index_x:.2f}, y={state.index_y:.2f} | Pointing: {state.is_pointing}",
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )
        y += line_height

    backend_label = backend if backend else "disabled"
    cv2.putText(frame, f"Backend: {backend_label}", (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Raspberry Pi hand gesture detector")

    parser.add_argument(
        "--camera",
        default=env_value("CAMERA_URL", "IP_CAMERA_URL", "VIDEO_SOURCE", default="0"),
        help="Camera index or IP camera URL. Defaults to CAMERA_URL/IP_CAMERA_URL/VIDEO_SOURCE from .env",
    )
    parser.add_argument(
        "--backend",
        default=None,
        help="Backend base URL, e.g. http://192.168.1.50:8000. Disabled by default for local Pi testing.",
    )

    parser.add_argument("--width", type=int, default=env_int("CAMERA_WIDTH", default=640))
    parser.add_argument("--height", type=int, default=env_int("CAMERA_HEIGHT", default=480))
    parser.add_argument("--fps", type=int, default=env_int("CAMERA_FPS", default=30))
    parser.add_argument(
        "--flip",
        action="store_true",
        default=False,
        help="Mirror camera image for more natural interaction.",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        default=False,
        help="Show debug camera window.",
    )

    parser.add_argument("--history-seconds", type=float, default=env_float("HAND_HISTORY_SECONDS", default=0.45))
    parser.add_argument("--swipe-min-dx", type=float, default=env_float("HAND_SWIPE_MIN_DX", default=0.22))
    parser.add_argument("--swipe-max-dy", type=float, default=env_float("HAND_SWIPE_MAX_DY", default=0.16))
    parser.add_argument("--pinch-threshold", type=float, default=env_float("HAND_PINCH_THRESHOLD", default=0.34))
    parser.add_argument(
        "--open-palm-hold-seconds",
        type=float,
        default=env_float("HAND_OPEN_PALM_HOLD_SECONDS", default=0.8),
    )
    parser.add_argument("--cooldown-seconds", type=float, default=env_float("HAND_COOLDOWN_SECONDS", default=1.0))
    parser.add_argument("--release-seconds", type=float, default=env_float("HAND_RELEASE_SECONDS", default=0.35))
    parser.add_argument("--swipe-rearm-dx", type=float, default=env_float("HAND_SWIPE_REARM_DX", default=0.10))

    parser.add_argument("--model-complexity", type=int, default=env_int("HAND_MODEL_COMPLEXITY", default=0), choices=[0, 1])
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=env_float("HAND_MIN_DETECTION_CONFIDENCE", default=0.65),
    )
    parser.add_argument(
        "--min-tracking-confidence",
        type=float,
        default=env_float("HAND_MIN_TRACKING_CONFIDENCE", default=0.65),
    )

    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    parsed_args = parser.parse_args()

    if ENV_FILE:
        print(f"Loaded .env from: {ENV_FILE}")
    else:
        print("No .env file found. Using CLI arguments and defaults.")

    print(f"Camera source: {parsed_args.camera}")
    print(f"Backend URL: {parsed_args.backend or 'disabled'}")

    run(parsed_args)

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
            # Safety: if eyes stay "closed" longer than long_blink_max, the
            # state machine is stuck (e.g. tracking lost mid-blink then resumed
            # with eyes already open). Reset so next blink can be detected.
            if now - self.blink_start_time > self.long_blink_max + 0.3:
                self.blink_closed = False
                self.close_streak = 0
                self.open_streak = 0

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
    thumb_pose: Optional[str] = None  # "thumbs_up" | "thumbs_down" | None


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


def _classify_thumb_pose(landmarks) -> Optional[str]:
    """Detect thumbs up/down.

    Finger-curl check uses distance from palm (rotation-invariant).
    Direction check uses image x-axis: DroidCam raw frame has head on the LEFT,
    so real-world UP = image LEFT = negative x.
    """
    wrist     = landmarks[0]
    thumb_mcp = landmarks[2]
    thumb_tip = landmarks[4]

    hand_scale = math.sqrt((wrist.x - landmarks[9].x) ** 2 + (wrist.y - landmarks[9].y) ** 2)
    if hand_scale < 1e-6:
        return None

    # Palm center (used as reference for finger curl check)
    palm_x = (wrist.x + landmarks[5].x + landmarks[9].x + landmarks[13].x + landmarks[17].x) / 5
    palm_y = (wrist.y + landmarks[5].y + landmarks[9].y + landmarks[13].y + landmarks[17].y) / 5

    # Non-thumb finger tips: must all be close to palm (curled)
    # Using distance from palm — rotation-invariant unlike y-axis comparison
    CURL_RATIO = 1.3  # tip must be within 1.3x hand_scale of palm
    finger_tips = [8, 12, 16, 20]
    curled = [
        math.sqrt((landmarks[i].x - palm_x) ** 2 + (landmarks[i].y - palm_y) ** 2) / hand_scale
        for i in finger_tips
    ]
    fingers_ok = all(d < CURL_RATIO for d in curled)

    # Thumb tip must be far from palm (extended)
    thumb_dist = math.sqrt((thumb_tip.x - palm_x) ** 2 + (thumb_tip.y - palm_y) ** 2) / hand_scale

    if not fingers_ok or thumb_dist < 0.4:
        return None

    # Direction via image x-axis — camera-rotation-aware.
    # DroidCam streams landscape from a portrait phone; enroll.py corrects it with
    # ROTATE_90_CLOCKWISE, meaning the raw frame has the person's head on the LEFT
    # (negative-x side). So: real-world UP = image LEFT = negative x.
    tx = thumb_tip.x - thumb_mcp.x
    ty = thumb_tip.y - thumb_mcp.y
    t_mag = math.sqrt(tx * tx + ty * ty)
    if t_mag < 1e-6:
        return None

    thumb_nx = tx / t_mag  # normalised x component of thumb direction

    if thumb_nx > 0.4:    # thumb points RIGHT in image = UP in real world
        return "thumbs_up"
    if thumb_nx < -0.4:   # thumb points LEFT in image = DOWN in real world
        return "thumbs_down"
    return None


class HandGestureDetector:
    def __init__(
        self,
        history_seconds: float = 0.45,
        swipe_min_dx: float = 0.22,
        swipe_max_dy: float = 0.16,
        pinch_threshold: float = 0.34,
        pinch_frames_needed: int = 4,
        open_palm_hold_seconds: float = 0.8,
        cooldown_seconds: float = 0.7,
        release_seconds: float = 0.25,
        swipe_rearm_dx: float = 0.10,
        swipe_min_duration: float = 0.12,
        post_swipe_pinch_block_seconds: float = 1.2,
        invert_swipe: bool = False,
    ) -> None:
        self.history_seconds = history_seconds
        self.swipe_min_dx = swipe_min_dx
        self.swipe_max_dy = swipe_max_dy
        self.pinch_threshold = pinch_threshold
        self.pinch_frames_needed = pinch_frames_needed
        self.open_palm_hold_seconds = open_palm_hold_seconds
        self.cooldown_seconds = cooldown_seconds
        self.release_seconds = release_seconds
        self.swipe_rearm_dx = swipe_rearm_dx
        self.swipe_min_duration = swipe_min_duration
        self.post_swipe_pinch_block_seconds = post_swipe_pinch_block_seconds
        self.invert_swipe = invert_swipe

        self.history: Deque[HandState] = deque()
        self.last_event_name: Optional[str] = None
        self.last_event_time: float = 0.0
        self.last_swipe_time: float = 0.0
        self.open_palm_start_time: Optional[float] = None
        self.was_pinching: bool = False
        self.pinch_streak: int = 0
        self.locked_event_name: Optional[str] = None
        self.release_start_time: Optional[float] = None
        self.locked_swipe_x: Optional[float] = None
        self._debug_log_time: float = 0.0

        # Thumbs up/down hold detection
        self.thumbs_hold_seconds: float = 0.6
        self._thumbs_held: Optional[str] = None   # current held pose name
        self._thumbs_start_time: Optional[float] = None

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

        # ring+pinky down is enough to distinguish from open palm (which needs 4+)
        is_pointing = (
            finger_state["index"]
            and not finger_state["ring"]
            and not finger_state["pinky"]
        )

        thumb_pose = _classify_thumb_pose(hand_landmarks.landmark)

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
            thumb_pose=thumb_pose,
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
        if event_name in {"swipe_left", "swipe_right"}:
            self.last_swipe_time = now
        self._lock_until_released(event_name, state)

        print(f"[GESTURE][EMIT] {event_name} ix={state.index_x:.3f} cx={state.center_x:.3f}")
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
        self.pinch_streak = 0
        self.locked_event_name = None
        self.release_start_time = None
        self.locked_swipe_x = None
        self._thumbs_held = None
        self._thumbs_start_time = None
        # Don't reset last_swipe_time — the post-swipe block should survive a
        # brief loss of hand detection (user may briefly leave the frame while
        # retracting after a swipe, which is exactly when false pinch fires).

    def _trim_history(self, now: float) -> None:
        while self.history and now - self.history[0].timestamp > self.history_seconds:
            self.history.popleft()

    def _detect_event(self, state: HandState) -> Optional[str]:
        if self._is_locked(state):
            return None

        # Pinch and pointing are mutually exclusive poses. During a swipe the
        # thumb can briefly cross close to the index tip (low pinch_ratio) for
        # a single frame — without this guard, that fires a false pinch and
        # cancels the swipe. Also require N consecutive frames so transient
        # finger motions don't register as a pinch.
        if state.is_pinching and not state.is_pointing:
            self.pinch_streak += 1
        else:
            self.pinch_streak = 0

        # Post-swipe pinch suppression: when the user retracts their hand
        # after a swipe, fingers naturally collapse toward a fist and pinch
        # ratio drops — this used to fire a phantom SELECT/BACK that cancelled
        # the navigation. Block pinch for a short window after every swipe.
        in_post_swipe_block = (
            state.timestamp - self.last_swipe_time < self.post_swipe_pinch_block_seconds
        )

        if (
            self.pinch_streak >= self.pinch_frames_needed
            and not self.was_pinching
            and not in_post_swipe_block
        ):
            self.was_pinching = True
            self.open_palm_start_time = None
            return "pinch"

        if not state.is_pinching:
            self.was_pinching = False

        # Suppress swipe while a thumb pose is active — hand movement during
        # thumbs up/down would otherwise fire phantom swipes.
        if state.thumb_pose is None:
            swipe_event = self._detect_pointing_swipe(state) or self._detect_freehand_swipe(state)
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

        # Thumbs up / thumbs down: hold pose for thumbs_hold_seconds to fire.
        if state.thumb_pose is not None:
            if state.thumb_pose == self._thumbs_held and self._thumbs_start_time is not None:
                if state.timestamp - self._thumbs_start_time >= self.thumbs_hold_seconds:
                    self._thumbs_start_time = state.timestamp  # reset for repeat
                    return state.thumb_pose
            else:
                self._thumbs_held = state.thumb_pose
                self._thumbs_start_time = state.timestamp
        else:
            self._thumbs_held = None
            self._thumbs_start_time = None

        return None

    def _detect_pointing_swipe(self, current: HandState) -> Optional[str]:
        """FPS-independent swipe detection using time-anchored displacement.

        Strategy: for each new pointing frame, search history for the OLDEST
        pointing sample whose age falls in [swipe_min_duration, history_seconds].
        Then check whether the index fingertip travelled enough horizontal
        distance with little vertical drift between that anchor and now.

        Why this works at any FPS:
          - Old algorithm needed >=4 samples in history; at 5 fps the 0.6s
            window held only 3 samples and swipe never fired.
          - This algorithm needs only ONE valid anchor + the current frame.
            At 5 fps that means ~2 samples — enough to measure dx/dy.
          - Choosing the OLDEST valid anchor maximises the displacement we
            can measure, so a slow-but-deliberate swipe still triggers.

        The min_duration floor (default 0.12s) rejects sub-frame jitter that
        would otherwise look like a very fast swipe. The history_seconds
        ceiling rejects "I held my finger here for a second then drifted"
        which is not a real swipe gesture.
        """
        _now = current.timestamp
        _throttle = _now - self._debug_log_time > 0.4  # max 2.5 log/sn

        if not current.is_pointing or current.is_pinching:
            if _throttle:
                self._debug_log_time = _now
                print(
                    f"[SWIPE] TEŞHİS — pointing={current.is_pointing} "
                    f"pinch={current.is_pinching} open_palm={current.is_open_palm} "
                    f"lock={self.locked_event_name} "
                    f"ix={current.index_x:.2f} iy={current.index_y:.2f}"
                )
            return None

        anchor: Optional[HandState] = None
        for sample in self.history:
            age = current.timestamp - sample.timestamp
            if age > self.history_seconds:
                continue  # too old, but keep scanning — history is chronological
            if age < self.swipe_min_duration:
                break  # too new; everything after is also too new
            if sample.is_pointing and not sample.is_pinching:
                anchor = sample
                break  # take the OLDEST valid one for max displacement window

        if anchor is None:
            if _throttle:
                self._debug_log_time = _now
                valid_hist = sum(
                    1 for s in self.history
                    if self.swipe_min_duration <= (_now - s.timestamp) <= self.history_seconds
                )
                print(
                    f"[SWIPE] TEŞHİS — anchor yok | history={len(self.history)} "
                    f"geçerli_pencere={valid_hist} "
                    f"(min={self.swipe_min_duration}s max={self.history_seconds}s)"
                )
            return None

        dx = current.index_x - anchor.index_x
        dy = current.index_y - anchor.index_y

        if abs(dy) > self.swipe_max_dy:
            print(f"[SWIPE] TEŞHİS — dikey hareket fazla: dy={dy:.3f} > {self.swipe_max_dy}")
            return None

        if dx <= -self.swipe_min_dx:
            self.history.clear()
            return self._map_swipe_event("swipe_left")

        if dx >= self.swipe_min_dx:
            self.history.clear()
            return self._map_swipe_event("swipe_right")

        if _throttle:
            self._debug_log_time = _now
            print(
                f"[SWIPE] TEŞHİS — hareket yetersiz: dx={dx:+.3f} "
                f"(gereken ±{self.swipe_min_dx})"
            )
        return None

    def _detect_freehand_swipe(self, current: HandState) -> Optional[str]:
        """El merkezi hareketine dayalı swipe — parmak yönünden bağımsız.

        Parmak kameraya doğru uzatılmış olsa bile (is_pointing=False) çalışır.
        Pinch ve open_palm_hold ile çakışmayı önlemek için:
        - Pinch aktifken devre dışı
        - Open palm aktifken devre dışı (open_palm_hold ile çakışır)
        """
        if current.is_pinching or current.is_open_palm:
            return None

        _now = current.timestamp
        anchor: Optional[HandState] = None
        for sample in self.history:
            age = _now - sample.timestamp
            if age > self.history_seconds:
                continue
            if age < self.swipe_min_duration:
                break
            if not sample.is_pinching and not sample.is_open_palm:
                anchor = sample
                break

        if anchor is None:
            return None

        dx = current.center_x - anchor.center_x
        dy = current.center_y - anchor.center_y

        if abs(dy) > self.swipe_max_dy:
            return None

        if dx <= -self.swipe_min_dx:
            self.history.clear()
            print(f"[SWIPE] freehand swipe_left dx={dx:.3f}")
            return self._map_swipe_event("swipe_left")

        if dx >= self.swipe_min_dx:
            self.history.clear()
            print(f"[SWIPE] freehand swipe_right dx={dx:.3f}")
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
        remaining = self.cooldown_seconds - (now - self.last_event_time)
        if remaining > 0:
            print(f"[GESTURE][COOLDOWN] suppress — {remaining:.3f}s kaldı (last={self.last_event_name})")
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
                print(f"[GESTURE][LOCK] '{self.locked_event_name}' release başladı")
            elif now - self.release_start_time >= self.release_seconds:
                print(f"[GESTURE][LOCK] '{self.locked_event_name}' lock temizlendi")
                self.locked_event_name = None
                self.release_start_time = None
                self.locked_swipe_x = None
                self.history.clear()
                return False
        else:
            if self.release_start_time is not None:
                print(f"[GESTURE][LOCK] '{self.locked_event_name}' release iptal — lock devam ediyor (pointing={state.is_pointing} ix={state.index_x:.3f} lock_x={self.locked_swipe_x})")
            self.release_start_time = None

        return True

    def _is_released_from_locked_event(self, state: HandState) -> bool:
        if self.locked_event_name == "pinch":
            return not state.is_pinching

        if self.locked_event_name == "open_palm_hold":
            return not state.is_open_palm

        if self.locked_event_name in ("thumbs_up", "thumbs_down"):
            return state.thumb_pose != self.locked_event_name

        if self.locked_event_name == "swipe_right":
            if self.locked_swipe_x is None:
                return not state.is_pointing
            return state.is_pointing and state.index_x <= self.locked_swipe_x - self.swipe_rearm_dx

        if self.locked_event_name == "swipe_left":
            if self.locked_swipe_x is None:
                return not state.is_pointing
            return state.is_pointing and state.index_x >= self.locked_swipe_x + self.swipe_rearm_dx

        return not state.is_pinching and not state.is_open_palm and not state.is_pointing
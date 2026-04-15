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
        self.ear_threshold = 0.21
        self.long_blink_ear = 0.18
        self.long_blink_min = 0.55
        self.long_blink_max = 1.5
        self.short_blink_min = 0.20
        self.short_blink_max = 0.35
        self.gaze_stable_time = 0.35

        self.last_gaze = None
        self.gaze_same_time = 0.0
        self.gaze_event_fired = False

        self.long_blink_closed = False
        self.long_blink_start_time = 0.0

        self.short_blink_closed = False
        self.short_blink_start_time = 0.0

    def process_face_landmarks(self, face_landmarks, frame_w: int, frame_h: int, now: float, dt: float):
        right_ear = calc_ear(face_landmarks, RIGHT_EYE_IDX, frame_w, frame_h)
        left_ear = calc_ear(face_landmarks, LEFT_EYE_IDX, frame_w, frame_h)
        ear = (right_ear + left_ear) / 2.0

        rx, ry = iris_relative_position(face_landmarks, RIGHT_IRIS_CENTER, RIGHT_EYE_IDX, frame_w, frame_h)
        lx, ly = iris_relative_position(face_landmarks, LEFT_IRIS_CENTER, LEFT_EYE_IDX, frame_w, frame_h)

        iris_x = (rx + lx) / 2.0
        iris_y = (ry + ly) / 2.0
        gaze_text = classify_gaze(iris_x, iris_y)

        if ear < self.long_blink_ear:
            if not self.long_blink_closed:
                self.long_blink_closed = True
                self.long_blink_start_time = now
        else:
            if self.long_blink_closed:
                duration = now - self.long_blink_start_time
                self.long_blink_closed = False
                if self.long_blink_min <= duration <= self.long_blink_max:
                    return build_event(
                        "long_blink",
                        metadata={"duration": duration, "ear": ear},
                        source="eye",
                    )

        if ear < self.ear_threshold:
            if not self.short_blink_closed:
                self.short_blink_closed = True
                self.short_blink_start_time = now
        else:
            if self.short_blink_closed:
                duration = now - self.short_blink_start_time
                self.short_blink_closed = False
                if self.short_blink_min <= duration <= self.short_blink_max:
                    return build_event(
                        "short_blink",
                        metadata={"duration": duration, "ear": ear},
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
        self.long_blink_closed = False
        self.short_blink_closed = False
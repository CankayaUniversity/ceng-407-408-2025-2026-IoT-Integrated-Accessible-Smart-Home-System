from events.models import build_event

TIP_IDS = {
    "thumb": 4,
    "index": 8,
    "middle": 12,
    "ring": 16,
    "pinky": 20,
}

PIP_IDS = {
    "index": 6,
    "middle": 10,
    "ring": 14,
    "pinky": 18,
}


class HandGestureDetector:
    def __init__(self):
        self.last_gesture = None
        self.same_count = 0
        self.required_frames = 5

    def _is_finger_up(self, landmarks, tip_id, pip_id):
        return landmarks[tip_id].y < landmarks[pip_id].y

    def _thumb_up(self, landmarks):
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        index_mcp = landmarks[5]

        thumb_is_up = thumb_tip.y < thumb_ip.y
        hand_is_vertical = thumb_tip.y < index_mcp.y
        return thumb_is_up and hand_is_vertical

    def classify(self, hand_landmarks):
        lm = hand_landmarks.landmark

        index_up = self._is_finger_up(lm, TIP_IDS["index"], PIP_IDS["index"])
        middle_up = self._is_finger_up(lm, TIP_IDS["middle"], PIP_IDS["middle"])
        ring_up = self._is_finger_up(lm, TIP_IDS["ring"], PIP_IDS["ring"])
        pinky_up = self._is_finger_up(lm, TIP_IDS["pinky"], PIP_IDS["pinky"])
        thumb_up = self._thumb_up(lm)

        up_count = sum([index_up, middle_up, ring_up, pinky_up])

        if up_count == 4:
            return "OPEN_PALM"
        if up_count == 0 and not thumb_up:
            return "FIST"
        if index_up and middle_up and not ring_up and not pinky_up:
            return "TWO_FINGERS"
        if thumb_up and not index_up and not middle_up and not ring_up and not pinky_up:
            return "THUMB_UP"

        return None

    def process_hand(self, hand_landmarks):
        gesture = self.classify(hand_landmarks)

        if gesture is None:
            self.last_gesture = None
            self.same_count = 0
            return None

        if gesture == self.last_gesture:
            self.same_count += 1
        else:
            self.last_gesture = gesture
            self.same_count = 1

        if self.same_count == self.required_frames:
            return build_event(
                event_name=gesture,
                metadata={"source_mode": "hand"},
                event_type="hand",
            )

        return None

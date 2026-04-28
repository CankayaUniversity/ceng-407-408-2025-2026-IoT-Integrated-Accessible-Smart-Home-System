import cv2
import mediapipe as mp

from config import (
    HAND_MIN_DETECTION_CONFIDENCE,
    HAND_MIN_TRACKING_CONFIDENCE,
    HAND_MODEL_COMPLEXITY,
)

mp_hands = mp.solutions.hands


class HandTracker:
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=HAND_MODEL_COMPLEXITY,
            min_detection_confidence=HAND_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=HAND_MIN_TRACKING_CONFIDENCE,
        )

    def process(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.hands.process(rgb)
        rgb.flags.writeable = True
        return results

    def close(self):
        self.hands.close()
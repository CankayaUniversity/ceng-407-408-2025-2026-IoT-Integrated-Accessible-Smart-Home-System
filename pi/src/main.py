import time
import threading
import queue

import cv2
import mediapipe as mp

from camera.ip_camera import IPCameraStream
from config import (
    BACKEND_EVENT_URL,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    IP_CAMERA_URL,
    SEND_EVENTS_TO_BACKEND,
    HAND_HISTORY_SECONDS,
    HAND_SWIPE_MIN_DX,
    HAND_SWIPE_MAX_DY,
    HAND_PINCH_THRESHOLD,
    HAND_OPEN_PALM_HOLD_SECONDS,
    HAND_COOLDOWN_SECONDS,
    HAND_RELEASE_SECONDS,
    HAND_SWIPE_REARM_DX,
    HAND_INVERT_SWIPE,
)
from events.publisher import publish_event
from integration.api_client import send_event
from vision.gesture_detector import GestureDetector, HandGestureDetector
from vision.hand_tracker import HandTracker

mp_face_mesh = mp.solutions.face_mesh

# Background queue for non-blocking event forwarding
_event_queue: queue.Queue = queue.Queue()


def _event_worker() -> None:
    """Background thread: picks events from queue and POSTs to backend."""
    while True:
        payload = _event_queue.get()
        if payload is None:
            break
        try:
            response = send_event(BACKEND_EVENT_URL, payload)
            if response is not None:
                print(f"[BACKEND] {response}")
        except Exception as exc:
            print(f"[BACKEND][ERR] {exc}")
        _event_queue.task_done()


def _forward_event(event) -> None:
    """
    Publish event locally and optionally forward it to backend.

    During Pi-only testing:
        SEND_EVENTS_TO_BACKEND=false

    During full integration:
        SEND_EVENTS_TO_BACKEND=true
    """
    payload = publish_event(event)

    if SEND_EVENTS_TO_BACKEND:
        _event_queue.put(payload)


def main():
    print("[SYSTEM] Starting Pi vision pipeline")
    print(f"[CONFIG] Camera URL: {IP_CAMERA_URL}")
    print(f"[CONFIG] Frame size: {FRAME_WIDTH}x{FRAME_HEIGHT}")
    print(f"[CONFIG] Backend sending: {SEND_EVENTS_TO_BACKEND}")
    print(f"[CONFIG] Backend URL: {BACKEND_EVENT_URL}")
    print(f"[CONFIG] Hand invert swipe: {HAND_INVERT_SWIPE}")

    camera = IPCameraStream(IP_CAMERA_URL, FRAME_WIDTH, FRAME_HEIGHT)
    if not camera.open():
        print("IP camera source could not be opened.")
        return

    worker = None
    if SEND_EVENTS_TO_BACKEND:
        worker = threading.Thread(target=_event_worker, daemon=True)
        worker.start()

    eye_detector = GestureDetector()
    hand_tracker = HandTracker()
    hand_detector = HandGestureDetector(
        history_seconds=HAND_HISTORY_SECONDS,
        swipe_min_dx=HAND_SWIPE_MIN_DX,
        swipe_max_dy=HAND_SWIPE_MAX_DY,
        pinch_threshold=HAND_PINCH_THRESHOLD,
        open_palm_hold_seconds=HAND_OPEN_PALM_HOLD_SECONDS,
        cooldown_seconds=HAND_COOLDOWN_SECONDS,
        release_seconds=HAND_RELEASE_SECONDS,
        swipe_rearm_dx=HAND_SWIPE_REARM_DX,
        invert_swipe=HAND_INVERT_SWIPE,
    )

    prev_time = time.time()
    frame_idx = 0

    try:
        with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as face_mesh:
            while True:
                ret, frame = camera.read()
                if not ret:
                    print("Frame could not be read.")
                    break

                now = time.time()
                dt = now - prev_time
                prev_time = now

                # Eye processing
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                face_results = face_mesh.process(rgb)
                rgb.flags.writeable = True

                if face_results.multi_face_landmarks:
                    face_landmarks = face_results.multi_face_landmarks[0].landmark
                    eye_event = eye_detector.process_face_landmarks(
                        face_landmarks, FRAME_WIDTH, FRAME_HEIGHT, now, dt
                    )
                    if eye_event is not None:
                        _forward_event(eye_event)
                else:
                    eye_detector.reset()

                # Hand processing
                hand_results = hand_tracker.process(frame)

                if hand_results.multi_hand_landmarks:
                    hand_landmarks = hand_results.multi_hand_landmarks[0]
                    hand_event = hand_detector.process_hand_landmarks(hand_landmarks, now)
                    if hand_event is not None:
                        _forward_event(hand_event)
                else:
                    hand_detector.reset()

                # Debug log every 20 frames
                frame_idx += 1
                if frame_idx % 20 == 0:
                    fps = 1.0 / dt if dt > 0 else 0
                    print(f"FPS:{fps:5.1f}  queue:{_event_queue.qsize()}")

                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break

    finally:
        if SEND_EVENTS_TO_BACKEND:
            _event_queue.put(None)

        hand_tracker.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
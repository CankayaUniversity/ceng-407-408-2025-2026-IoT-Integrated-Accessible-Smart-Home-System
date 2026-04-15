import time

import cv2
import mediapipe as mp

from camera.ip_camera import IPCameraStream
from config import BACKEND_EVENT_URL, FRAME_HEIGHT, FRAME_WIDTH, IP_CAMERA_URL
from events.publisher import publish_event
from integration.api_client import send_event
from vision.gesture_detector import GestureDetector

mp_face_mesh = mp.solutions.face_mesh


def _forward_event(event) -> None:
    payload = publish_event(event)
    response = send_event(BACKEND_EVENT_URL, payload)
    if response is not None:
        print(f"[BACKEND] {response}")


def main():
    camera = IPCameraStream(IP_CAMERA_URL, FRAME_WIDTH, FRAME_HEIGHT)
    if not camera.open():
        print("IP camera source could not be opened.")
        return

    eye_detector = GestureDetector()
    prev_time = time.time()

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

            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
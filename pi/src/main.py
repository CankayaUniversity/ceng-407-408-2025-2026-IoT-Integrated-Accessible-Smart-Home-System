import time
import cv2
import mediapipe as mp

from camera.ip_camera import IPCameraStream
from config import IP_CAMERA_URL, FRAME_WIDTH, FRAME_HEIGHT, BACKEND_EVENT_URL
from events.publisher import publish_event
from integration.api_client import send_event
from mapping.command_mapper import map_event_to_command
from vision.gesture_detector import GestureDetector
from vision.hand_tracker import HandTracker
from vision.hand_gesture_detector import HandGestureDetector

mp_face_mesh = mp.solutions.face_mesh


def main():
    camera = IPCameraStream(IP_CAMERA_URL, FRAME_WIDTH, FRAME_HEIGHT)
    if not camera.open():
        print("IP camera source could not be opened.")
        return

    eye_detector = GestureDetector()
    hand_tracker = HandTracker()
    hand_detector = HandGestureDetector()

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

            # -------------------------
            # Eye / face pipeline
            # -------------------------
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
                    payload = publish_event(eye_event)
                    command = map_event_to_command(eye_event.event_name)
                    print(f"[COMMAND] {command}")

                    # Backend hazır olunca açarsın
                    # send_event(BACKEND_EVENT_URL, payload)
            else:
                eye_detector.reset()

            # -------------------------
            # Hand pipeline
            # -------------------------
            hand_results = hand_tracker.process(frame)

            if hand_results.multi_hand_landmarks:
                hand_event = hand_detector.process_hand(hand_results.multi_hand_landmarks[0])

                if hand_event is not None:
                    payload = publish_event(hand_event)
                    command = map_event_to_command(hand_event.event_name)
                    print(f"[COMMAND] {command}")

                    # Backend hazır olunca açarsın
                    # send_event(BACKEND_EVENT_URL, payload)

            # İstersen görüntüyü görmek için aç
            # cv2.imshow("Pi Vision Control", frame)

            # key = cv2.waitKey(1) & 0xFF
            # if key in (27, ord("q")):
            #    break
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break

    hand_tracker.close()
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
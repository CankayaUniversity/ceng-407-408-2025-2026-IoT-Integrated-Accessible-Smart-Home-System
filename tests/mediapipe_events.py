import cv2
import time
import numpy as np
import mediapipe as mp
from collections import deque

# ================== CONFIG ==================
# IP_CAMERA_URL = "http://192.x.x.x:4747/video"
IP_CAMERA_URL = "http://10.x.x.x:4747/video"

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# EAR / blink parameters
EAR_THRESHOLD = 0.21       # blink reference threshold (eyes closed)
LONG_BLINK_EAR = 0.18      # stricter threshold for long blink
MIN_EAR_FRAMES = 2         # (kept for reference; not strictly required in this script)

# Long blink duration window (seconds)
LONG_BLINK_MIN = 0.55
LONG_BLINK_MAX = 1.5

# Short blink duration window (seconds)
SHORT_BLINK_MAX = 0.35
SHORT_BLINK_MIN = 0.2

# Gaze tolerance
CENTER_TOL = 0.1           # horizontal tolerance around center
UP_DOWN_SCALE = 1.6        # vertical tolerance scale (y is usually noisier)

# Gaze event parameters
GAZE_STABLE_TIME = 0.35    # if LEFT/RIGHT stays this long -> fire an event

# FPS smoothing
fps_hist = deque(maxlen=20)

# On-screen event text display time (seconds)
EVENT_DISPLAY_TIME = 1.2

# ============== MEDIAPIPE SETTINGS ===========
mp_face_mesh = mp.solutions.face_mesh

RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]
LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]

RIGHT_IRIS_CENTER = 468
LEFT_IRIS_CENTER  = 473


# ================== HELPER FUNCTIONS ==================
def euclidean(p1, p2):
    return np.linalg.norm(p1 - p2)


def calc_ear(landmarks, eye_indices, img_w, img_h):
    pts = np.array(
        [[landmarks[i].x * img_w, landmarks[i].y * img_h] for i in eye_indices],
        dtype=np.float32,
    )
    p1, p2, p3, p4, p5, p6 = pts
    denom = 2.0 * euclidean(p1, p4)
    if denom == 0:
        return 1.0
    return float((euclidean(p2, p6) + euclidean(p3, p5)) / denom)


def iris_relative_position(landmarks, iris_idx, eye_indices, img_w, img_h):
    iris = np.array(
        [landmarks[iris_idx].x * img_w, landmarks[iris_idx].y * img_h],
        dtype=np.float32,
    )

    left_corner = np.array(
        [landmarks[eye_indices[0]].x * img_w, landmarks[eye_indices[0]].y * img_h],
        dtype=np.float32,
    )
    right_corner = np.array(
        [landmarks[eye_indices[3]].x * img_w, landmarks[eye_indices[3]].y * img_h],
        dtype=np.float32,
    )

    top = np.mean(
        [
            [landmarks[eye_indices[1]].x * img_w, landmarks[eye_indices[1]].y * img_h],
            [landmarks[eye_indices[2]].x * img_w, landmarks[eye_indices[2]].y * img_h],
        ],
        axis=0,
        dtype=np.float32,
    )
    bottom = np.mean(
        [
            [landmarks[eye_indices[4]].x * img_w, landmarks[eye_indices[4]].y * img_h],
            [landmarks[eye_indices[5]].x * img_w, landmarks[eye_indices[5]].y * img_h],
        ],
        axis=0,
        dtype=np.float32,
    )

    eye_width = euclidean(left_corner, right_corner)
    eye_height = euclidean(top, bottom)

    if eye_width <= 0 or eye_height <= 0:
        return 0.5, 0.5

    x_rel = (iris[0] - left_corner[0]) / eye_width
    y_rel = (iris[1] - top[1]) / eye_height

    x_rel = float(np.clip(x_rel, 0.0, 1.0))
    y_rel = float(np.clip(y_rel, 0.0, 1.0))
    return x_rel, y_rel


def classify_gaze(x_rel, y_rel):
    dx = x_rel - 0.5
    dy = y_rel - 0.5

    center_tol_x = CENTER_TOL
    center_tol_y = CENTER_TOL * UP_DOWN_SCALE

    
    if abs(dx) < center_tol_x and abs(dy) < center_tol_y:
        return "CENTER"

    
    if abs(dx) > abs(dy):
        # mirrored camera preview: left right flipped our current setup(front camera)
        return "LEFT" if dx > 0 else "RIGHT"
    else:
        return "DOWN" if dy > 0 else "UP"


# ================== MAIN LOOP ==================
def main():
    cap = cv2.VideoCapture(IP_CAMERA_URL)
    if not cap.isOpened():
        print("IP camera source could not be opened.")
        return

    prev_time = time.time()
    frame_idx = 0

    # Gaze event state
    last_gaze = None
    gaze_same_time = 0.0
    gaze_event_fired = False

    # Long blink state
    long_blink_closed = False
    long_blink_start_time = 0.0

    # Short blink state
    short_blink_closed = False
    short_blink_start_time = 0.0

    # Last event shown on screen
    last_event_text = ""
    last_event_time = 0.0

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame could not be read (stream ended or connection issue).")
                break

            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            h, w, _ = frame.shape

            # FPS computation (smoothed)
            curr = time.time()
            dt = curr - prev_time
            prev_time = curr

            fps_inst = (1.0 / dt) if dt > 0 else 0.0
            if 0 < fps_inst < 60:
                fps_hist.append(fps_inst)
            fps = (sum(fps_hist) / len(fps_hist)) if fps_hist else fps_inst

            # MediaPipe inference
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = face_mesh.process(rgb)
            rgb.flags.writeable = True

            gaze_text = "NO_FACE"
            ear_value = None

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0].landmark

                # EAR
                right_ear = calc_ear(face_landmarks, RIGHT_EYE_IDX, w, h)
                left_ear = calc_ear(face_landmarks, LEFT_EYE_IDX, w, h)
                ear = (right_ear + left_ear) / 2.0
                ear_value = ear

                # Iris + gaze
                rx, ry = iris_relative_position(face_landmarks, RIGHT_IRIS_CENTER, RIGHT_EYE_IDX, w, h)
                lx, ly = iris_relative_position(face_landmarks, LEFT_IRIS_CENTER, LEFT_EYE_IDX, w, h)

                iris_x = (rx + lx) / 2.0
                iris_y = (ry + ly) / 2.0

                gaze_text = classify_gaze(iris_x, iris_y)

                # --------- LONG BLINK STATE MACHINE ---------
                if ear < LONG_BLINK_EAR:
                    if not long_blink_closed:
                        long_blink_closed = True
                        long_blink_start_time = curr
                else:
                    if long_blink_closed:
                        duration = curr - long_blink_start_time
                        long_blink_closed = False
                        if LONG_BLINK_MIN <= duration <= LONG_BLINK_MAX:
                            msg = f"LONG_BLINK ({duration:.2f}s)"
                            print(f">>> EVENT: {msg}")
                            last_event_text = msg
                            last_event_time = curr
                            # TODO: trigger command here later

                # --------- SHORT BLINK STATE MACHINE ---------
                if ear < EAR_THRESHOLD:
                    if not short_blink_closed:
                        short_blink_closed = True
                        short_blink_start_time = curr
                else:
                    if short_blink_closed:
                        duration = curr - short_blink_start_time
                        short_blink_closed = False
                        if SHORT_BLINK_MIN <= duration <= SHORT_BLINK_MAX:
                            msg = f"SHORT_BLINK ({duration:.2f}s)"
                            print(f">>> EVENT: {msg}")
                            last_event_text = msg
                            last_event_time = curr
                            # TODO: trigger command here later

                # --------- GAZE LEFT / RIGHT STATE MACHINE ---------
                if gaze_text in ("LEFT", "RIGHT"):
                    if gaze_text == last_gaze:
                        gaze_same_time += dt
                    else:
                        last_gaze = gaze_text
                        gaze_same_time = dt
                        gaze_event_fired = False
                else:
                    # Reset when CENTER / UP / DOWN / NO_FACE
                    last_gaze = gaze_text
                    gaze_same_time = 0.0
                    gaze_event_fired = False

                if (
                    gaze_text in ("LEFT", "RIGHT")
                    and not gaze_event_fired
                    and gaze_same_time >= GAZE_STABLE_TIME
                ):
                    msg = f"LOOK_{gaze_text}"
                    print(f">>> EVENT: {msg}")
                    last_event_text = msg
                    last_event_time = curr
                    gaze_event_fired = True

            else:
                # If face is lost, reset all states
                long_blink_closed = False
                short_blink_closed = False
                last_gaze = "NO_FACE"
                gaze_same_time = 0.0
                gaze_event_fired = False

            # ============ VISUAL OVERLAY ============
            cv2.putText(frame, f"FPS: {fps:4.1f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"GAZE: {gaze_text}", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            ear_dbg = f"{ear_value:.3f}" if ear_value is not None else "---"
            cv2.putText(frame, f"EAR: {ear_dbg}", (10, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Show the last fired event for a short time
            if last_event_text and (curr - last_event_time) < EVENT_DISPLAY_TIME:
                cv2.putText(frame, last_event_text,
                            (int(FRAME_WIDTH * 0.20), int(FRAME_HEIGHT * 0.12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # Debug log (every 20 frames)
            frame_idx += 1
            if frame_idx % 20 == 0:
                print(f"FPS:{fps:5.1f}  GAZE:{gaze_text:7s}  EAR:{ear_dbg}")

            cv2.imshow("MediaPipe Events (GUI) - Pi5 PoC", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):  # ESC or q
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

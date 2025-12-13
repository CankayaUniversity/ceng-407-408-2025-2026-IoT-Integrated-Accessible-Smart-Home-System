import cv2
import time
import numpy as np
import mediapipe as mp
from collections import deque

# ================== settings ==================
IP_CAMERA_URL = "http://10.x.x.x:4747/video"


FRAME_WIDTH = 640
FRAME_HEIGHT = 480

EAR_THRESHOLD = 0.21
MIN_EAR_FRAMES = 2

CENTER_TOL = 0.1       # horizontal tolerance for LEFT/RIGHT
UP_DOWN_SCALE = 1.6    # vertical tolerance

# FPS smoothing
fps_hist = deque(maxlen=20)

# ============== MEDIAPIPE settings ===========
mp_face_mesh = mp.solutions.face_mesh

RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]
LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]

RIGHT_IRIS_CENTER = 468
LEFT_IRIS_CENTER  = 473


# ================== helper functions ==================
def euclidean(p1, p2):
    return np.linalg.norm(p1 - p2)

def calc_ear(landmarks, eye_indices, img_w, img_h):
    pts = np.array(
        [[landmarks[i].x * img_w, landmarks[i].y * img_h] for i in eye_indices],
        dtype=np.float32,
    )
    p1, p2, p3, p4, p5, p6 = pts
    return (euclidean(p2, p6) + euclidean(p3, p5)) / (2.0 * euclidean(p1, p4))

def iris_relative_position(landmarks, iris_idx, eye_indices, img_w, img_h):
    iris = np.array([landmarks[iris_idx].x * img_w, landmarks[iris_idx].y * img_h])

    left_corner = np.array([landmarks[eye_indices[0]].x * img_w,
                            landmarks[eye_indices[0]].y * img_h])
    right_corner = np.array([landmarks[eye_indices[3]].x * img_w,
                             landmarks[eye_indices[3]].y * img_h])

    top_pts = [
        np.array([landmarks[eye_indices[1]].x * img_w,
                  landmarks[eye_indices[1]].y * img_h]),
        np.array([landmarks[eye_indices[2]].x * img_w,
                  landmarks[eye_indices[2]].y * img_h])
    ]
    bottom_pts = [
        np.array([landmarks[eye_indices[4]].x * img_w,
                  landmarks[eye_indices[4]].y * img_h]),
        np.array([landmarks[eye_indices[5]].x * img_w,
                  landmarks[eye_indices[5]].y * img_h])
    ]

    top = np.mean(top_pts, axis=0)
    bottom = np.mean(bottom_pts, axis=0)

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
    center_tol_y = CENTER_TOL * UP_DOWN_SCALE  # UP/DOWN tollerance

    # CENTER 
    if abs(dx) < center_tol_x and abs(dy) < center_tol_y:
        return "CENTER"

    # 
    if abs(dx) > abs(dy):
        return "LEFT" if dx > 0 else "RIGHT" # mirrored camera preview: left right flipped our current setup(front camera)
    else:
        return "DOWN" if dy > 0 else "UP"


# ================== MAIN LOOP ==================
def main():

    cap = cv2.VideoCapture(IP_CAMERA_URL)
    if not cap.isOpened():
        print("IP camera source could not be opened.")
        return

    blink_frame_counter = 0
    prev_time = time.time()
    frame_idx = 0

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame could not be read.")
                break

            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            h, w, _ = frame.shape

            # FPS calc
            curr = time.time()
            dt = curr - prev_time
            prev_time = curr

            fps_inst = 1.0 / dt if dt > 0 else 0
            if 0 < fps_inst < 60:
                fps_hist.append(fps_inst)

            fps = sum(fps_hist)/len(fps_hist) if fps_hist else fps_inst

            # MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = face_mesh.process(rgb)
            rgb.flags.writeable = True

            blink_text = ""
            gaze_text = "NO_FACE"
            ear_value = None

            if results.multi_face_landmarks:

                face_landmarks = results.multi_face_landmarks[0].landmark

                # EAR
                right_ear = calc_ear(face_landmarks, RIGHT_EYE_IDX, w, h)
                left_ear = calc_ear(face_landmarks, LEFT_EYE_IDX, w, h)
                ear = (right_ear + left_ear) / 2.0
                ear_value = ear

                if ear < EAR_THRESHOLD:
                    blink_frame_counter += 1
                else:
                    if blink_frame_counter >= MIN_EAR_FRAMES:
                        blink_text = "BLINK"
                    blink_frame_counter = 0

                # Gaze
                rx, ry = iris_relative_position(face_landmarks, RIGHT_IRIS_CENTER, RIGHT_EYE_IDX, w, h)
                lx, ly = iris_relative_position(face_landmarks, LEFT_IRIS_CENTER, LEFT_EYE_IDX, w, h)

                iris_x = (rx + lx) / 2.0
                iris_y = (ry + ly) / 2.0

                gaze_text = classify_gaze(iris_x, iris_y)

            # Terminal log (every frame)
            print(f"FPS:{fps:5.1f}  GAZE:{gaze_text:7s}  BLINK:{blink_text:5s}  EAR:{ear_value if ear_value else 0:.3f}")

    cap.release()


if __name__ == "__main__":
    main()

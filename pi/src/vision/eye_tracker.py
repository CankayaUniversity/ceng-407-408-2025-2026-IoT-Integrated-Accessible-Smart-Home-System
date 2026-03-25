import numpy as np

RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]
LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]

RIGHT_IRIS_CENTER = 468
LEFT_IRIS_CENTER = 473

CENTER_TOL = 0.1
UP_DOWN_SCALE = 1.6


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

    return float(np.clip(x_rel, 0.0, 1.0)), float(np.clip(y_rel, 0.0, 1.0))


def classify_gaze(x_rel, y_rel):
    dx = x_rel - 0.5
    dy = y_rel - 0.5

    center_tol_x = CENTER_TOL
    center_tol_y = CENTER_TOL * UP_DOWN_SCALE

    if abs(dx) < center_tol_x and abs(dy) < center_tol_y:
        return "CENTER"

    if abs(dx) > abs(dy):
        return "LEFT" if dx > 0 else "RIGHT"
    return "DOWN" if dy > 0 else "UP"
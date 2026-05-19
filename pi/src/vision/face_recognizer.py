import json
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


class FaceRecognizer:
    """OpenCV LBPH-based face recognizer for access control.

    Loads a pre-trained model from disk. If no model exists (no enrolled users),
    is_trained is False and the caller should treat all users as authorized.
    """

    CONFIDENCE_THRESHOLD = 52  # lower = stricter match; 0-100 scale (LBPH)

    def __init__(self, data_dir: str = "face_data") -> None:
        self._data_dir = Path(data_dir)
        self._model_path = self._data_dir / "model.yml"
        self._labels_path = self._data_dir / "labels.json"
        self._cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

        self._recognizer = cv2.face.LBPHFaceRecognizer_create()
        self._detector = cv2.CascadeClassifier(self._cascade_path)
        self._labels: dict[int, str] = {}

        self.is_trained = self._load()

    def _load(self) -> bool:
        if not self._model_path.exists() or not self._labels_path.exists():
            return False
        try:
            self._recognizer.read(str(self._model_path))
            with open(self._labels_path) as f:
                raw = json.load(f)
            self._labels = {int(k): v for k, v in raw.items()}
            print(f"[FACE] Model yüklendi: {list(self._labels.values())}")
            return True
        except Exception as exc:
            print(f"[FACE] Model yüklenemedi: {exc}")
            return False

    def _prepare_gray(self, frame: np.ndarray) -> np.ndarray:
        """Convert to grayscale and rotate to compensate for sideways camera."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)

    def _detect_face_roi(self, gray: np.ndarray) -> Optional[np.ndarray]:
        """Return the largest detected face ROI or None."""
        faces = self._detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(40, 40),
        )
        if len(faces) == 0:
            return None
        # Pick the largest face
        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
        return gray[y : y + h, x : x + w]

    def recognize(self, frame: np.ndarray) -> tuple[Optional[str], float]:
        """Recognize a face in the frame.

        Returns:
            (user_name, confidence) if an authorized face is detected.
            (None, confidence) if unknown face or no face detected.
        """
        gray = self._prepare_gray(frame)
        roi = self._detect_face_roi(gray)

        if roi is None:
            return None, 999.0  # no face in frame

        roi_resized = cv2.resize(roi, (200, 200))
        roi_eq = cv2.equalizeHist(roi_resized)
        label_id, confidence = self._recognizer.predict(roi_eq)

        if confidence < self.CONFIDENCE_THRESHOLD:
            user = self._labels.get(label_id, "unknown")
            return user, confidence
        return None, confidence


def train(data_dir: str = "face_data") -> None:
    """Train LBPH model from enrolled face images saved by enroll.py."""
    data_path = Path(data_dir)
    labels_path = data_path / "labels.json"

    if not labels_path.exists():
        print("[FACE] Kayıtlı kullanıcı bulunamadı.")
        return

    with open(labels_path) as f:
        labels: dict[str, str] = json.load(f)  # {str(id): name}

    images, label_ids = [], []
    for label_id_str, name in labels.items():
        user_dir = data_path / name
        for img_path in sorted(user_dir.glob("*.jpg")):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(cv2.equalizeHist(cv2.resize(img, (200, 200))))
                label_ids.append(int(label_id_str))

    if not images:
        print("[FACE] Hiçbir yüz görüntüsü bulunamadı.")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(images, np.array(label_ids))
    recognizer.save(str(data_path / "model.yml"))
    print(f"[FACE] Model eğitildi: {len(images)} görüntü, {len(labels)} kullanıcı")

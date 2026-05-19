"""
Yüz kayıt scripti — yeni kullanıcı ekler ve LBPH modelini yeniden eğitir.

Kullanım:
    cd pi/src
    python3 enroll.py --name "Ömer"

Kameradan 40 kare yakalar, her birinde yüz bulur ve face_data/<name>/ klasörüne kaydeder.
Sonra LBPH modelini yeniden eğitir.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

# config'den kamera URL'ini almak için src/ dizinine ekle
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import IP_CAMERA_URL, FRAME_WIDTH, FRAME_HEIGHT
from vision.face_recognizer import train

DATA_DIR = Path(__file__).resolve().parent.parent / "face_data"
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
CAPTURE_COUNT = 60
CAPTURE_INTERVAL = 0.15  # seconds between captures


def _next_label_id(labels: dict[str, str]) -> int:
    if not labels:
        return 0
    return max(int(k) for k in labels) + 1


def enroll(name: str) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    user_dir = DATA_DIR / name
    user_dir.mkdir(exist_ok=True)

    labels_path = DATA_DIR / "labels.json"
    if labels_path.exists():
        with open(labels_path) as f:
            labels: dict[str, str] = json.load(f)
    else:
        labels = {}

    # Assign label id (reuse if user already exists)
    label_id = next((int(k) for k, v in labels.items() if v == name), None)
    if label_id is None:
        label_id = _next_label_id(labels)
        labels[str(label_id)] = name

    detector = cv2.CascadeClassifier(CASCADE_PATH)
    cap = cv2.VideoCapture(IP_CAMERA_URL)
    if not cap.isOpened():
        print(f"Kamera açılamadı: {IP_CAMERA_URL}")
        sys.exit(1)

    print(f"\n[KAYIT] '{name}' için yüz kaydı başlıyor.")
    print(f"[KAYIT] Kameraya bakın. {CAPTURE_COUNT} kare kaydedilecek...")
    print("[KAYIT] Başlamak için Enter'a basın...")
    input()

    saved = 0
    existing = len(list(user_dir.glob("*.jpg")))

    while saved < CAPTURE_COUNT:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
        faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(40, 40))

        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
            roi = gray[y : y + h, x : x + w]
            roi_resized = cv2.resize(roi, (200, 200))
            img_path = user_dir / f"{existing + saved:04d}.jpg"
            cv2.imwrite(str(img_path), roi_resized)
            saved += 1
            print(f"[KAYIT] {saved}/{CAPTURE_COUNT} kare kaydedildi", end="\r")
        else:
            print(f"[KAYIT] Yüz bulunamadı, {saved}/{CAPTURE_COUNT}", end="\r")

        time.sleep(CAPTURE_INTERVAL)

    cap.release()
    print(f"\n[KAYIT] {saved} kare kaydedildi → {user_dir}")

    with open(labels_path, "w") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    print("[KAYIT] Model eğitiliyor...")
    train(str(DATA_DIR))
    print(f"[KAYIT] ✓ '{name}' başarıyla eklendi.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Yüz kayıt aracı")
    parser.add_argument("--name", required=True, help="Kullanıcı adı (örn: Ömer)")
    args = parser.parse_args()

    name = args.name.strip()
    if not name:
        print("Hata: Kullanıcı adı boş olamaz.")
        sys.exit(1)

    enroll(name)


if __name__ == "__main__":
    main()

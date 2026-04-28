import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from pi/ root (one level up from src/)
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)


def _bool_env(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).lower() in {"1", "true", "yes", "y", "on"}


IP_CAMERA_URL = os.getenv(
    "IP_CAMERA_URL",
    os.getenv("CAMERA_URL", "http://192.168.1.36:4747/video"),
)

BACKEND_EVENT_URL = os.getenv(
    "BACKEND_EVENT_URL",
    "http://127.0.0.1:8000/vision-events",
)

SEND_EVENTS_TO_BACKEND = _bool_env("SEND_EVENTS_TO_BACKEND", "false")

FRAME_WIDTH = int(os.getenv("FRAME_WIDTH", os.getenv("CAMERA_WIDTH", "640")))
FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", os.getenv("CAMERA_HEIGHT", "480")))

# Hand gesture settings
HAND_HISTORY_SECONDS = float(os.getenv("HAND_HISTORY_SECONDS", "0.45"))
HAND_SWIPE_MIN_DX = float(os.getenv("HAND_SWIPE_MIN_DX", "0.22"))
HAND_SWIPE_MAX_DY = float(os.getenv("HAND_SWIPE_MAX_DY", "0.16"))
HAND_PINCH_THRESHOLD = float(os.getenv("HAND_PINCH_THRESHOLD", "0.34"))
HAND_OPEN_PALM_HOLD_SECONDS = float(os.getenv("HAND_OPEN_PALM_HOLD_SECONDS", "0.8"))
HAND_COOLDOWN_SECONDS = float(os.getenv("HAND_COOLDOWN_SECONDS", "0.7"))
HAND_RELEASE_SECONDS = float(os.getenv("HAND_RELEASE_SECONDS", "0.25"))
HAND_SWIPE_REARM_DX = float(os.getenv("HAND_SWIPE_REARM_DX", "0.10"))
HAND_INVERT_SWIPE = _bool_env("HAND_INVERT_SWIPE", "false")

HAND_MODEL_COMPLEXITY = int(os.getenv("HAND_MODEL_COMPLEXITY", "0"))
HAND_MIN_DETECTION_CONFIDENCE = float(os.getenv("HAND_MIN_DETECTION_CONFIDENCE", "0.65"))
HAND_MIN_TRACKING_CONFIDENCE = float(os.getenv("HAND_MIN_TRACKING_CONFIDENCE", "0.65"))
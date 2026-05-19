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

# Derive control-mode URL from event URL by default so users only configure one
# host. They can still override explicitly if backend is split across services.
def _derive_control_mode_url(event_url: str) -> str:
    base = event_url.rsplit("/", 1)[0]
    return f"{base}/control-mode"

BACKEND_CONTROL_MODE_URL = os.getenv(
    "BACKEND_CONTROL_MODE_URL",
    _derive_control_mode_url(BACKEND_EVENT_URL),
)

CONTROL_MODE_POLL_SECONDS = float(os.getenv("CONTROL_MODE_POLL_SECONDS", "2.0"))

SEND_EVENTS_TO_BACKEND = _bool_env("SEND_EVENTS_TO_BACKEND", "false")

FRAME_WIDTH = int(os.getenv("FRAME_WIDTH", os.getenv("CAMERA_WIDTH", "480")))
FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", os.getenv("CAMERA_HEIGHT", "360")))

# Hand gesture settings
HAND_HISTORY_SECONDS = float(os.getenv("HAND_HISTORY_SECONDS", "0.6"))
HAND_SWIPE_MIN_DX = float(os.getenv("HAND_SWIPE_MIN_DX", "0.18"))
HAND_SWIPE_MAX_DY = float(os.getenv("HAND_SWIPE_MAX_DY", "0.18"))
HAND_PINCH_THRESHOLD = float(os.getenv("HAND_PINCH_THRESHOLD", "0.34"))
HAND_PINCH_FRAMES_NEEDED = int(os.getenv("HAND_PINCH_FRAMES_NEEDED", "4"))
HAND_POST_SWIPE_PINCH_BLOCK_SECONDS = float(os.getenv("HAND_POST_SWIPE_PINCH_BLOCK_SECONDS", "1.2"))
HAND_OPEN_PALM_HOLD_SECONDS = float(os.getenv("HAND_OPEN_PALM_HOLD_SECONDS", "0.8"))
HAND_COOLDOWN_SECONDS = float(os.getenv("HAND_COOLDOWN_SECONDS", "0.7"))
HAND_RELEASE_SECONDS = float(os.getenv("HAND_RELEASE_SECONDS", "0.25"))
HAND_SWIPE_REARM_DX = float(os.getenv("HAND_SWIPE_REARM_DX", "0.10"))
HAND_SWIPE_MIN_DURATION = float(os.getenv("HAND_SWIPE_MIN_DURATION", "0.12"))
HAND_INVERT_SWIPE = _bool_env("HAND_INVERT_SWIPE", "false")

HAND_MODEL_COMPLEXITY = int(os.getenv("HAND_MODEL_COMPLEXITY", "0"))
HAND_MIN_DETECTION_CONFIDENCE = float(os.getenv("HAND_MIN_DETECTION_CONFIDENCE", "0.65"))
HAND_MIN_TRACKING_CONFIDENCE = float(os.getenv("HAND_MIN_TRACKING_CONFIDENCE", "0.65"))

# Face-based access control
FACE_AUTH_REQUIRED = _bool_env("FACE_AUTH_REQUIRED", "false")
FACE_RECOGNITION_INTERVAL = float(os.getenv("FACE_RECOGNITION_INTERVAL", "2.0"))
FACE_AUTH_GRACE_PERIOD = float(os.getenv("FACE_AUTH_GRACE_PERIOD", "3.0"))
FACE_DATA_PATH = os.getenv("FACE_DATA_PATH", str(Path(__file__).resolve().parent.parent / "face_data"))
FACE_AUTH_URL = os.getenv(
    "FACE_AUTH_URL",
    BACKEND_EVENT_URL.rsplit("/", 1)[0] + "/face-auth",
)
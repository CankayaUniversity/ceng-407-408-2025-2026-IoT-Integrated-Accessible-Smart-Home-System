import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from pi/ root (one level up from src/)
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)

IP_CAMERA_URL = os.getenv("IP_CAMERA_URL", "http://192.168.1.36:4747/video")
BACKEND_EVENT_URL = os.getenv("BACKEND_EVENT_URL", "http://127.0.0.1:8000/vision-events")
FRAME_WIDTH = int(os.getenv("FRAME_WIDTH", "640"))
FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", "480"))
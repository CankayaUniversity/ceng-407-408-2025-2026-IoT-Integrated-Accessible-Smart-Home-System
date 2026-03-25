import os

IP_CAMERA_URL = os.getenv("IP_CAMERA_URL", "http://10.x.x.x:4747/video")
BACKEND_EVENT_URL = os.getenv("BACKEND_EVENT_URL", "http://127.0.0.1:8000/api/events")

FRAME_WIDTH = int(os.getenv("FRAME_WIDTH", "640"))
FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", "480"))
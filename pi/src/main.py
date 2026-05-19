"""
Pi vision pipeline — multi-threaded.

Architecture
------------
Sequential per-frame processing (camera read -> face mesh -> hands) saturates
a single core and caps the pipeline at ~10 FPS on a Pi 5. MediaPipe releases
the GIL during native inference, so running the two models on separate threads
gives true parallelism on Pi 5's four cores.

Threads
-------
1. Camera thread          : reads frames into a single-slot LatestFrameBuffer
2. Eye worker thread      : pulls latest frame, runs face mesh, emits eye events
3. Hand worker thread     : pulls latest frame, runs hands, emits hand events
4. Backend forwarder      : drains the event queue and POSTs to the API
5. Main thread            : starts/stops workers, reports per-worker FPS

Frame sharing
-------------
LatestFrameBuffer is a single-slot, overwrite-on-write buffer. Workers track
the last frame_id they processed and only re-run inference when a NEWER frame
is available. This avoids two failure modes:
  - Backlog: a queue would let frames pile up if a worker is slow, so the
    pipeline would process stale frames instead of the freshest one.
  - Wasted CPU: without frame_id tracking, a fast worker would re-process the
    same frame back-to-back while waiting for the camera.
"""

import signal
import threading
import time
import queue
from typing import Optional

import cv2
import mediapipe as mp

from camera.ip_camera import IPCameraStream
import requests

from config import (
    BACKEND_CONTROL_MODE_URL,
    BACKEND_EVENT_URL,
    CONTROL_MODE_POLL_SECONDS,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    IP_CAMERA_URL,
    SEND_EVENTS_TO_BACKEND,
    HAND_HISTORY_SECONDS,
    HAND_SWIPE_MIN_DX,
    HAND_SWIPE_MAX_DY,
    HAND_PINCH_THRESHOLD,
    HAND_PINCH_FRAMES_NEEDED,
    HAND_OPEN_PALM_HOLD_SECONDS,
    HAND_COOLDOWN_SECONDS,
    HAND_RELEASE_SECONDS,
    HAND_SWIPE_REARM_DX,
    HAND_SWIPE_MIN_DURATION,
    HAND_POST_SWIPE_PINCH_BLOCK_SECONDS,
    HAND_INVERT_SWIPE,
    FACE_AUTH_REQUIRED,
    FACE_RECOGNITION_INTERVAL,
    FACE_AUTH_GRACE_PERIOD,
    FACE_DATA_PATH,
    FACE_AUTH_URL,
    BACKEND_EVENT_URL,
)
from events.publisher import publish_event
from integration.api_client import fetch_control_mode, send_event
from vision.face_recognizer import FaceRecognizer
from vision.gesture_detector import GestureDetector, HandGestureDetector
from vision.hand_tracker import HandTracker

mp_face_mesh = mp.solutions.face_mesh

# Thread-safe queue drained by the backend forwarder thread.
_event_queue: queue.Queue = queue.Queue()

# Set when the user requests shutdown (Ctrl+C or SIGTERM). All loops poll this.
_stop_event = threading.Event()

# Per-source enable flags driven by the mode poller. Workers gate their
# expensive MediaPipe inference behind these — when a source is disabled it
# sits in a 100ms sleep loop instead of burning CPU on inference whose result
# the backend would just throw away. Both default to set() so the pipeline
# starts processing immediately, before the first poll completes; the poller
# clears whichever one the backend says is disabled within ~2 seconds.
_eye_enabled = threading.Event()
_eye_enabled.set()
_hand_enabled = threading.Event()
_hand_enabled.set()

# Face-based access control state.
# _face_authorized is set when a known face is detected, cleared otherwise.
# _last_authorized_time supports a grace period so a brief frame dropout
# doesn't immediately block gesture commands.
_face_authorized = threading.Event()
_face_authorized.set()  # default: authorized until face loop decides otherwise
_last_authorized_time: float = 0.0


# =========================================================================
# Latest-frame buffer (single-slot, overwrite-on-write)
# =========================================================================

class LatestFrameBuffer:
    """Single-slot frame buffer shared by the camera producer and N consumers.

    The producer overwrites the slot on every new frame. Consumers track the
    last frame_id they processed and only consume when a NEWER frame is
    available — re-running MediaPipe on the same frame would waste CPU and
    re-emitting a queue would build latency.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame = None
        self._frame_id = 0

    def put(self, frame) -> None:
        with self._lock:
            # Copy so the producer can safely overwrite its own buffer next
            # iteration without racing with a consumer that is still reading.
            self._frame = frame.copy()
            self._frame_id += 1

    def get_if_newer(self, last_seen_id: int):
        """Return (frame_copy, new_id) or (None, last_seen_id) if nothing new."""
        with self._lock:
            if self._frame is None or self._frame_id == last_seen_id:
                return None, last_seen_id
            return self._frame, self._frame_id


# =========================================================================
# FPS counter (per-thread)
# =========================================================================

class FPSCounter:
    """Tracks FPS for the calling thread; logs every `report_every` seconds."""

    def __init__(self, label: str, report_every: float = 2.0) -> None:
        self.label = label
        self.report_every = report_every
        self._count = 0
        self._last_report = time.time()
        self._fps = 0.0

    def tick(self) -> None:
        self._count += 1
        now = time.time()
        elapsed = now - self._last_report
        if elapsed >= self.report_every:
            self._fps = self._count / elapsed
            print(f"[FPS][{self.label}] {self._fps:5.1f}  queue:{_event_queue.qsize()}")
            self._count = 0
            self._last_report = now


# =========================================================================
# Worker functions
# =========================================================================

def _backend_forwarder() -> None:
    """Drains the event queue and POSTs to backend. Runs until queue gets None."""
    while True:
        payload = _event_queue.get()
        if payload is None:
            break
        t_send = time.time()
        print(f"[BACKEND][SEND] {payload['source']}:{payload['name']} @ {t_send:.3f}")
        try:
            response = send_event(BACKEND_EVENT_URL, payload)
            elapsed = time.time() - t_send
            if response is not None:
                ui = response.get("ui_state", {})
                action = response.get("mapped_action") or response.get("action")
                print(f"[BACKEND][OK] {elapsed*1000:.0f}ms | action={action} | screen={ui.get('current_screen')} idx={ui.get('selected_index')} item={ui.get('selected_item')}")
        except Exception as exc:
            print(f"[BACKEND][ERR] {exc}")
        _event_queue.task_done()


def _mode_poller() -> None:
    """Polls backend control_mode and toggles per-source enable flags.

    A transient network error keeps the previous mode active — we never want
    to silently disable both sources because a single GET timed out.
    """
    last_mode: str | None = None
    while not _stop_event.is_set():
        data = fetch_control_mode(BACKEND_CONTROL_MODE_URL)
        if data is not None:
            sources = data.get("enabled_sources") or {}
            eye_on = bool(sources.get("eye", True))
            hand_on = bool(sources.get("hand", True))

            if eye_on:
                _eye_enabled.set()
            else:
                _eye_enabled.clear()
            if hand_on:
                _hand_enabled.set()
            else:
                _hand_enabled.clear()

            mode = data.get("control_mode")
            if mode != last_mode:
                print(f"[MODE] control_mode={mode} eye={eye_on} hand={hand_on}")
                last_mode = mode

        _stop_event.wait(CONTROL_MODE_POLL_SECONDS)


def _post_face_auth(status: str, user: Optional[str] = None, confidence: Optional[float] = None) -> None:
    """Notify backend about face auth state change. Fire-and-forget."""
    if not SEND_EVENTS_TO_BACKEND:
        return
    try:
        requests.post(
            FACE_AUTH_URL,
            json={"status": status, "user": user, "confidence": confidence},
            timeout=2.0,
        )
    except Exception:
        pass


def _forward_event(event) -> None:
    # Block gesture events when face auth is active and no known face has been authorized.
    # Authorization is only revoked when a recognized-but-unknown face appears (not on "no face").
    if FACE_AUTH_REQUIRED and not _face_authorized.is_set():
        print(f"[FACE] Yetkisiz — event bloklandı: {event.source}:{event.name}")
        return

    payload = publish_event(event)
    if SEND_EVENTS_TO_BACKEND:
        qsize = _event_queue.qsize()
        print(f"[QUEUE] kuyruğa eklendi: {payload['source']}:{payload['name']} (kuyruk boyutu: {qsize + 1})")
        _event_queue.put(payload)


def _face_loop(frame_buffer: LatestFrameBuffer) -> None:
    """Runs face recognition at a low frequency to authorize/block gesture events."""
    global _last_authorized_time

    if not FACE_AUTH_REQUIRED:
        _face_authorized.set()
        print("[FACE] FACE_AUTH_REQUIRED=false — yüz tanıma devre dışı, hep yetkili")
        return

    recognizer = FaceRecognizer(FACE_DATA_PATH)
    if not recognizer.is_trained:
        _face_authorized.set()
        print("[FACE] Kayıtlı yüz yok — hep yetkili (önce enroll.py çalıştırın)")
        return

    _face_authorized.clear()  # start locked; first recognition will unlock
    print(f"[FACE] Erişim kontrolü aktif — {FACE_RECOGNITION_INTERVAL}s aralıkla taranıyor")

    last_id = 0
    last_check = 0.0
    # Consecutive-vote tracking: only acts after 2 agreeing results in a row.
    # "No face in frame" (confidence > 900) resets the vote and keeps current state.
    prev_result: Optional[str] = "__init__"
    consecutive = 0

    while not _stop_event.is_set():
        now = time.time()
        if now - last_check < FACE_RECOGNITION_INTERVAL:
            time.sleep(0.1)
            continue

        frame, last_id = frame_buffer.get_if_newer(last_id)
        if frame is None:
            time.sleep(0.1)
            continue

        user, confidence = recognizer.recognize(frame)
        last_check = time.time()

        # No face in frame — keep current auth state, reset consecutive counter
        if confidence > 900:
            consecutive = 0
            prev_result = "__init__"
            continue

        # Face present: track consecutive identical results
        if user == prev_result:
            consecutive += 1
        else:
            consecutive = 1
            prev_result = user

        print(f"[FACE] tanıma: {user or 'yetkisiz'} güven={confidence:.1f} ardışık={consecutive}")

        # Only act after 2 consecutive agreeing results (reduces single-frame false positives)
        if consecutive < 2:
            continue

        if user:
            was_blocked = not _face_authorized.is_set()
            _face_authorized.set()
            _last_authorized_time = time.time()
            if was_blocked:
                print(f"[FACE] ✓ Yetkili: {user} (güven: {confidence:.1f})")
                _post_face_auth("authorized", user, round(confidence, 1))
        else:
            was_authorized = _face_authorized.is_set()
            _face_authorized.clear()
            if was_authorized:
                print(f"[FACE] ✗ Yetkisiz — tanınmayan yüz (güven: {confidence:.1f})")
                _post_face_auth("unauthorized")


def _enroll_loop(frame_buffer: LatestFrameBuffer) -> None:
    """Polls backend for enrollment requests and captures face frames."""
    if not SEND_EVENTS_TO_BACKEND:
        return

    base_url = BACKEND_EVENT_URL.rsplit("/", 1)[0]
    enroll_url = f"{base_url}/enroll-status"
    update_url = f"{base_url}/enroll-update"
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def post_update(status: str, progress: int = 0, total: int = 40, message: Optional[str] = None):
        try:
            requests.post(update_url, json={
                "status": status, "progress": progress,
                "total": total, "message": message,
            }, timeout=3)
        except Exception:
            pass

    last_id = 0

    while not _stop_event.is_set():
        try:
            resp = requests.get(enroll_url, timeout=3)
            state = resp.json()
        except Exception:
            _stop_event.wait(3)
            continue

        if state.get("status") != "requested":
            _stop_event.wait(2)
            continue

        name = state.get("name", "").strip()
        if not name:
            _stop_event.wait(2)
            continue

        print(f"[ENROLL] '{name}' için kayıt başladı")
        import json as _json
        from pathlib import Path as _Path
        from vision.face_recognizer import train as _train

        data_dir = _Path(FACE_DATA_PATH)
        user_dir = data_dir / name
        user_dir.mkdir(parents=True, exist_ok=True)

        labels_path = data_dir / "labels.json"
        if labels_path.exists():
            with open(labels_path) as f:
                labels = _json.load(f)
        else:
            labels = {}

        label_id = next((int(k) for k, v in labels.items() if v == name), None)
        if label_id is None:
            label_id = (max(int(k) for k in labels) + 1) if labels else 0
            labels[str(label_id)] = name

        existing = len(list(user_dir.glob("*.jpg")))
        saved = 0
        total = 60
        consecutive_miss = 0

        post_update("capturing", 0, total, "Kameraya bakın...")

        while saved < total and not _stop_event.is_set():
            frame, last_id = frame_buffer.get_if_newer(last_id)
            if frame is None:
                time.sleep(0.05)
                continue

            gray = cv2.rotate(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.ROTATE_90_CLOCKWISE)
            faces = cascade.detectMultiScale(gray, 1.1, 3, minSize=(40, 40))

            if len(faces) == 0:
                consecutive_miss += 1
                if consecutive_miss % 20 == 0:
                    post_update("capturing", saved, total, "Yüz bulunamadı, kameraya bakın...")
                time.sleep(0.1)
                continue

            consecutive_miss = 0
            x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
            roi = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
            cv2.imwrite(str(user_dir / f"{existing + saved:04d}.jpg"), roi)
            saved += 1
            post_update("capturing", saved, total, f"Yüz kaydediliyor: {saved}/{total}")
            time.sleep(0.1)

        if saved < total:
            post_update("error", saved, total, "Kayıt iptal edildi.")
            print(f"[ENROLL] İptal — yalnızca {saved}/{total} kare kaydedildi")
            continue

        post_update("training", saved, total, "Model eğitiliyor...")
        print("[ENROLL] Model eğitiliyor...")

        with open(labels_path, "w") as f:
            _json.dump(labels, f, ensure_ascii=False, indent=2)

        try:
            _train(str(data_dir))
            post_update("done", total, total, f"'{name}' başarıyla kaydedildi!")
            print(f"[ENROLL] ✓ '{name}' kaydedildi")
        except Exception as exc:
            post_update("error", 0, total, f"Eğitim hatası: {exc}")
            print(f"[ENROLL] Eğitim hatası: {exc}")


def _camera_loop(camera: IPCameraStream, frame_buffer: LatestFrameBuffer) -> None:
    fps = FPSCounter("camera")
    while not _stop_event.is_set():
        ret, frame = camera.read()
        if not ret:
            # Brief sleep on read failure so we don't busy-loop on a dead stream.
            time.sleep(0.05)
            continue
        frame_buffer.put(frame)
        fps.tick()


def _eye_loop(frame_buffer: LatestFrameBuffer) -> None:
    eye_detector = GestureDetector()
    fps = FPSCounter("eye")
    last_id = 0
    prev_time = time.time()

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.3,  # lower → tracking survives closed eyes
    ) as face_mesh:
        no_face_frames = 0
        NO_FACE_RESET_THRESHOLD = 6  # consecutive missed frames before reset

        while not _stop_event.is_set():
            if not _eye_enabled.is_set():
                # Backend has eye source disabled — skip inference entirely so
                # the active source (hand) gets all the CPU. Reset detector
                # state so we don't accumulate stale gaze/blink history while
                # paused.
                eye_detector.reset()
                no_face_frames = 0
                prev_time = time.time()
                time.sleep(0.1)
                continue

            frame, last_id = frame_buffer.get_if_newer(last_id)
            if frame is None:
                # No new frame yet; sleep briefly to avoid burning a core.
                time.sleep(0.005)
                continue

            now = time.time()
            dt = now - prev_time
            prev_time = now

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            face_results = face_mesh.process(rgb)
            rgb.flags.writeable = True

            if face_results.multi_face_landmarks:
                no_face_frames = 0
                face_landmarks = face_results.multi_face_landmarks[0].landmark
                eye_event = eye_detector.process_face_landmarks(
                    face_landmarks, FRAME_WIDTH, FRAME_HEIGHT, now, dt
                )
                if eye_event is not None:
                    _forward_event(eye_event)
            else:
                # Don't reset immediately — MediaPipe can miss a few frames
                # during a blink. Only reset after sustained face loss.
                no_face_frames += 1
                if no_face_frames >= NO_FACE_RESET_THRESHOLD:
                    eye_detector.reset()

            fps.tick()


def _hand_loop(frame_buffer: LatestFrameBuffer) -> None:
    hand_tracker = HandTracker()
    hand_detector = HandGestureDetector(
        history_seconds=HAND_HISTORY_SECONDS,
        swipe_min_dx=HAND_SWIPE_MIN_DX,
        swipe_max_dy=HAND_SWIPE_MAX_DY,
        pinch_threshold=HAND_PINCH_THRESHOLD,
        pinch_frames_needed=HAND_PINCH_FRAMES_NEEDED,
        open_palm_hold_seconds=HAND_OPEN_PALM_HOLD_SECONDS,
        cooldown_seconds=HAND_COOLDOWN_SECONDS,
        release_seconds=HAND_RELEASE_SECONDS,
        swipe_rearm_dx=HAND_SWIPE_REARM_DX,
        swipe_min_duration=HAND_SWIPE_MIN_DURATION,
        post_swipe_pinch_block_seconds=HAND_POST_SWIPE_PINCH_BLOCK_SECONDS,
        invert_swipe=HAND_INVERT_SWIPE,
    )
    fps = FPSCounter("hand")
    last_id = 0

    try:
        while not _stop_event.is_set():
            if not _hand_enabled.is_set():
                # Backend has hand source disabled — skip inference. Reset
                # detector state so swipe history doesn't leak across a pause.
                hand_detector.reset()
                time.sleep(0.1)
                continue

            frame, last_id = frame_buffer.get_if_newer(last_id)
            if frame is None:
                time.sleep(0.005)
                continue

            now = time.time()
            hand_results = hand_tracker.process(frame)

            if hand_results.multi_hand_landmarks:
                hand_landmarks = hand_results.multi_hand_landmarks[0]
                hand_event = hand_detector.process_hand_landmarks(hand_landmarks, now)
                if hand_event is not None:
                    _forward_event(hand_event)
            else:
                hand_detector.reset()

            fps.tick()
    finally:
        hand_tracker.close()


# =========================================================================
# Orchestration
# =========================================================================

def _install_signal_handlers() -> None:
    def handler(signum, frame):
        print(f"\n[SYSTEM] Signal {signum} received, shutting down...")
        _stop_event.set()

    signal.signal(signal.SIGINT, handler)
    try:
        signal.signal(signal.SIGTERM, handler)
    except (AttributeError, ValueError):
        # SIGTERM is not available on Windows when not in main thread.
        pass


def main():
    print("[SYSTEM] Starting Pi vision pipeline (multi-threaded)")
    print(f"[CONFIG] Camera URL: {IP_CAMERA_URL}")
    print(f"[CONFIG] Frame size: {FRAME_WIDTH}x{FRAME_HEIGHT}")
    print(f"[CONFIG] Backend sending: {SEND_EVENTS_TO_BACKEND}")
    print(f"[CONFIG] Backend URL: {BACKEND_EVENT_URL}")
    print(f"[CONFIG] Hand invert swipe: {HAND_INVERT_SWIPE}")
    print(f"[CONFIG] Face auth required: {FACE_AUTH_REQUIRED}")

    camera = IPCameraStream(IP_CAMERA_URL, FRAME_WIDTH, FRAME_HEIGHT)
    if not camera.open():
        print("IP camera source could not be opened.")
        return

    _install_signal_handlers()

    frame_buffer = LatestFrameBuffer()

    threads = [
        threading.Thread(target=_camera_loop, args=(camera, frame_buffer), name="camera", daemon=True),
        threading.Thread(target=_eye_loop, args=(frame_buffer,), name="eye", daemon=True),
        threading.Thread(target=_hand_loop, args=(frame_buffer,), name="hand", daemon=True),
        threading.Thread(target=_face_loop, args=(frame_buffer,), name="face", daemon=True),
        threading.Thread(target=_enroll_loop, args=(frame_buffer,), name="enroll", daemon=True),
    ]

    forwarder: Optional[threading.Thread] = None
    mode_poller: Optional[threading.Thread] = None
    if SEND_EVENTS_TO_BACKEND:
        forwarder = threading.Thread(target=_backend_forwarder, name="backend", daemon=True)
        forwarder.start()

        mode_poller = threading.Thread(target=_mode_poller, name="mode_poller", daemon=True)
        mode_poller.start()

    for t in threads:
        t.start()

    try:
        # Main thread idles until shutdown is requested.
        while not _stop_event.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        _stop_event.set()
    finally:
        print("[SYSTEM] Stopping workers...")
        _stop_event.set()
        for t in threads:
            t.join(timeout=2.0)

        if forwarder is not None:
            _event_queue.put(None)
            forwarder.join(timeout=2.0)

        if mode_poller is not None:
            mode_poller.join(timeout=2.0)

        camera.release()
        cv2.destroyAllWindows()
        print("[SYSTEM] Shutdown complete")


if __name__ == "__main__":
    main()

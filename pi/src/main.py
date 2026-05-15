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
)
from events.publisher import publish_event
from integration.api_client import fetch_control_mode, send_event
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


def _forward_event(event) -> None:
    payload = publish_event(event)
    if SEND_EVENTS_TO_BACKEND:
        qsize = _event_queue.qsize()
        print(f"[QUEUE] kuyruğa eklendi: {payload['source']}:{payload['name']} (kuyruk boyutu: {qsize + 1})")
        _event_queue.put(payload)


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
        min_tracking_confidence=0.5,
    ) as face_mesh:
        while not _stop_event.is_set():
            if not _eye_enabled.is_set():
                # Backend has eye source disabled — skip inference entirely so
                # the active source (hand) gets all the CPU. Reset detector
                # state so we don't accumulate stale gaze/blink history while
                # paused.
                eye_detector.reset()
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
                face_landmarks = face_results.multi_face_landmarks[0].landmark
                eye_event = eye_detector.process_face_landmarks(
                    face_landmarks, FRAME_WIDTH, FRAME_HEIGHT, now, dt
                )
                if eye_event is not None:
                    _forward_event(eye_event)
            else:
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

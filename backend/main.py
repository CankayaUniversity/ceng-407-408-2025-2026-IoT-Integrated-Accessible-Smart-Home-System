import json
import time as _time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

class Feedback(BaseModel):
    message: str
    severity: str = Field(..., examples=["success", "error", "warning", "info"])
    speak: bool

try:
    from device_controller import SmartHomeDeviceController
except ImportError:  # pragma: no cover
    from .device_controller import SmartHomeDeviceController  # type: ignore


app = FastAPI(
    title="Accessible Smart Home Modular API",
    version="1.2.0",
    description=(
        "Eye/hand vision events -> control mode filtering -> "
        "user-defined mappings -> UI action execution -> smart device control."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# Request / Response Models
# =========================

class VisionEventRequest(BaseModel):
    source: str = Field(..., examples=["eye", "hand"])
    name: str = Field(
        ...,
        examples=[
            "look_left",
            "look_right",
            "short_blink",
            "long_blink",
            "swipe_left",
            "swipe_right",
            "pinch",
            "open_palm_hold",
        ],
    )
    confidence: float | None = None
    timestamp: float | str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MappingUpdateRequest(BaseModel):
    mappings: dict[str, str]


class ActionExecuteRequest(BaseModel):
    action: str = Field(..., examples=["NAV_LEFT", "SELECT", "CONFIRM", "LIGHT_ON"])


class ControlModeUpdateRequest(BaseModel):
    control_mode: str = Field(..., examples=["eye_only", "hand_only"])


class FaceAuthRequest(BaseModel):
    status: str = Field(..., examples=["authorized", "unauthorized"])
    user: str | None = None
    confidence: float | None = None


class LightColorRequest(BaseModel):
    hue: int | None = None
    saturation: int | None = None
    brightness: int = 80
    color_temperature: int | None = None


class EnrollRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=50)


class EnrollUpdateRequest(BaseModel):
    status: str  # "capturing" | "training" | "done" | "error"
    progress: int = 0
    total: int = 40
    message: str | None = None


# =========================
# Controller
# =========================

controller = SmartHomeDeviceController()


# =========================
# Supported Inputs / Actions
# =========================

SUPPORTED_EVENT_SOURCES = {
    "eye",
    "hand",
}

SUPPORTED_CONTROL_MODES = {
    "eye_only",
    "hand_only",
}

SUPPORTED_VISION_EVENTS = {
    # Eye events
    "eye:look_left",
    "eye:look_right",
    "eye:short_blink",
    "eye:long_blink",

    # Hand events
    "hand:swipe_left",
    "hand:swipe_right",
    "hand:pinch",
    "hand:open_palm_hold",
    "hand:thumbs_up",
    "hand:thumbs_down",
}

SUPPORTED_ACTIONS = {
    "NAV_LEFT",
    "NAV_RIGHT",
    "SELECT",
    "CONFIRM",
    "BACK",
    "LIGHT_ON",
    "LIGHT_OFF",
    "PLUG_ON",
    "PLUG_OFF",
}

DEVICE_ACTIONS = {
    "LIGHT_ON",
    "LIGHT_OFF",
    "PLUG_ON",
    "PLUG_OFF",
}


# Default user-editable mapping
#
# Format:
#   source:name -> action
#
# Examples:
#   eye:look_left       -> NAV_LEFT
#   hand:swipe_right    -> NAV_RIGHT
#   hand:pinch          -> SELECT
mapping_store: dict[str, str] = {
    "eye:look_left": "NAV_LEFT",
    "eye:look_right": "NAV_RIGHT",
    "eye:short_blink": "LIGHT_ON",
    "eye:long_blink": "LIGHT_OFF",

    "hand:swipe_left": "NAV_LEFT",
    "hand:swipe_right": "NAV_RIGHT",
    "hand:pinch": "SELECT",
    "hand:open_palm_hold": "BACK",
    "hand:thumbs_up": "NAV_RIGHT",
    "hand:thumbs_down": "NAV_LEFT",
}


# =========================
# System / UI State
# =========================

system_state: dict[str, Any] = {
    "last_vision_event": None,
    "last_ignored_vision_event": None,
    "last_action": None,
    "last_command": None,

    # Control mode state
    # eye_only  -> only eye events are processed
    # hand_only -> only hand events are processed
    "control_mode": "eye_only",
    "enabled_sources": {
        "eye": True,
        "hand": False,
    },

    "device_status": {
        "light": "off",
        "plug": "off",
    },

    "face_auth": {
        "status": "unknown",   # "authorized" | "unauthorized" | "unknown"
        "user": None,
        "confidence": None,
        "updated_at": None,
    },
}

FACE_DATA_PATH = Path(__file__).resolve().parent.parent / "pi" / "face_data"

enroll_state: dict[str, Any] = {
    "status": "idle",       # "idle" | "requested" | "capturing" | "training" | "done" | "error"
    "name": None,
    "progress": 0,
    "total": 40,
    "message": None,
    "updated_at": None,
}

ui_state: dict[str, Any] = {
    "current_screen": "main_menu",
    "selected_index": 0,
    "screens": {
        "main_menu": ["Light", "Plug", "Status"],
        "light_menu": ["On", "Off", "Back"],
        "plug_menu": ["On", "Off", "Back"],
        "status_menu": ["Refresh", "Back"],
    },
}


# =========================
# Helpers
# =========================

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_vision_event_name(raw: str) -> str:
    return raw.strip().lower()


def normalize_event_source(raw: str) -> str:
    return raw.strip().lower()


def normalize_action_name(raw: str) -> str:
    return raw.strip().upper()


def normalize_control_mode(raw: str) -> str:
    return raw.strip().lower()


def build_event_key(source: str, name: str) -> str:
    normalized_source = normalize_event_source(source)
    normalized_name = normalize_vision_event_name(name)
    return f"{normalized_source}:{normalized_name}"


def legacy_event_key(name: str) -> str:
    """
    Backward compatibility for old clients that only sent eye event names.

    Example:
        look_left -> eye:look_left
    """
    normalized_name = normalize_vision_event_name(name)
    return f"eye:{normalized_name}"


def apply_control_mode(control_mode: str) -> None:
    control_mode = normalize_control_mode(control_mode)

    if control_mode not in SUPPORTED_CONTROL_MODES:
        raise HTTPException(status_code=400, detail=f"Unsupported control mode: {control_mode}")

    if control_mode == "eye_only":
        system_state["enabled_sources"] = {
            "eye": True,
            "hand": False,
        }

    elif control_mode == "hand_only":
        system_state["enabled_sources"] = {
            "eye": False,
            "hand": True,
        }

    system_state["control_mode"] = control_mode


def is_source_enabled(source: str) -> bool:
    enabled_sources = system_state.get("enabled_sources", {})
    return bool(enabled_sources.get(source, False))


def get_current_items() -> list[str]:
    return ui_state["screens"][ui_state["current_screen"]]


def get_selected_item() -> str:
    items = get_current_items()
    index = ui_state["selected_index"]
    return items[index]


def build_ui_snapshot() -> dict[str, Any]:
    return {
        "current_screen": ui_state["current_screen"],
        "selected_index": ui_state["selected_index"],
        "selected_item": get_selected_item(),
        "items": get_current_items(),
    }


def go_main_menu() -> None:
    ui_state["current_screen"] = "main_menu"
    ui_state["selected_index"] = 0


def handle_back() -> None:
    if ui_state["current_screen"] == "main_menu":
        return
    go_main_menu()


# =========================
# Device Execution
# =========================

async def execute_device_command(command: str) -> dict[str, Any]:
    command = normalize_action_name(command)

    if command not in DEVICE_ACTIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported device command: {command}")

    feedback_msg = "Komut uygulandı"
    if command == "LIGHT_ON":
        system_state["device_status"]["light"] = "on"
        feedback_msg = "Işık açıldı"
    elif command == "LIGHT_OFF":
        system_state["device_status"]["light"] = "off"
        feedback_msg = "Işık kapandı"
    elif command == "PLUG_ON":
        system_state["device_status"]["plug"] = "on"
        feedback_msg = "Priz açıldı"
    elif command == "PLUG_OFF":
        system_state["device_status"]["plug"] = "off"
        feedback_msg = "Priz kapandı"

    try:
        hardware_result = await controller.execute_async(command)
    except Exception as exc:
        raise HTTPException(
            status_code=502, 
            detail={
                "error": f"Device execution failed: {exc}",
                "feedback": {
                    "message": "Cihaz komutu başarısız",
                    "severity": "error",
                    "speak": True
                }
            }
        ) from exc

    system_state["last_command"] = {
        "command": command,
        "timestamp": now_iso(),
        "hardware_result": hardware_result,
    }

    return {
        "success": True,
        "executed_command": command,
        "device_status": system_state["device_status"],
        "hardware_result": hardware_result,
        "feedback": {
            "message": feedback_msg,
            "severity": "success",
            "speak": True
        }
    }


# =========================
# UI Action Handling
# =========================

async def handle_select() -> dict[str, Any]:
    selected = get_selected_item()
    screen = ui_state["current_screen"]

    if screen == "main_menu":
        if selected == "Light":
            ui_state["current_screen"] = "light_menu"
            ui_state["selected_index"] = 0
            return {
                "success": True,
                "action": "OPEN_MENU",
                "target": "light_menu",
                "ui_state": build_ui_snapshot(),
            }

        if selected == "Plug":
            ui_state["current_screen"] = "plug_menu"
            ui_state["selected_index"] = 0
            return {
                "success": True,
                "action": "OPEN_MENU",
                "target": "plug_menu",
                "ui_state": build_ui_snapshot(),
            }

        if selected == "Status":
            ui_state["current_screen"] = "status_menu"
            ui_state["selected_index"] = 0
            return {
                "success": True,
                "action": "OPEN_MENU",
                "target": "status_menu",
                "ui_state": build_ui_snapshot(),
            }

    if screen == "light_menu":
        if selected == "On":
            result = await execute_device_command("LIGHT_ON")
            return {
                "success": True,
                "action": "DEVICE_COMMAND",
                "result": result,
                "ui_state": build_ui_snapshot(),
                "feedback": result.get("feedback"),
            }

        if selected == "Off":
            result = await execute_device_command("LIGHT_OFF")
            return {
                "success": True,
                "action": "DEVICE_COMMAND",
                "result": result,
                "ui_state": build_ui_snapshot(),
                "feedback": result.get("feedback"),
            }

        if selected == "Back":
            handle_back()
            return {
                "success": True,
                "action": "BACK",
                "ui_state": build_ui_snapshot(),
            }

    if screen == "plug_menu":
        if selected == "On":
            result = await execute_device_command("PLUG_ON")
            return {
                "success": True,
                "action": "DEVICE_COMMAND",
                "result": result,
                "ui_state": build_ui_snapshot(),
                "feedback": result.get("feedback"),
            }

        if selected == "Off":
            result = await execute_device_command("PLUG_OFF")
            return {
                "success": True,
                "action": "DEVICE_COMMAND",
                "result": result,
                "ui_state": build_ui_snapshot(),
                "feedback": result.get("feedback"),
            }

        if selected == "Back":
            handle_back()
            return {
                "success": True,
                "action": "BACK",
                "ui_state": build_ui_snapshot(),
            }

    if screen == "status_menu":
        if selected == "Refresh":
            return {
                "success": True,
                "action": "REFRESH_STATUS",
                "device_status": system_state["device_status"],
                "ui_state": build_ui_snapshot(),
            }

        if selected == "Back":
            handle_back()
            return {
                "success": True,
                "action": "BACK",
                "ui_state": build_ui_snapshot(),
            }

    return {
        "success": True,
        "action": "NO_OP",
        "ui_state": build_ui_snapshot(),
    }


async def execute_action(action: str, trigger: str = "manual", trigger_detail: str | None = None) -> dict[str, Any]:
    action = normalize_action_name(action)

    if action not in SUPPORTED_ACTIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported action: {action}")

    if action == "NAV_LEFT":
        items = get_current_items()
        max_index = len(items) - 1
        ui_state["selected_index"] = max_index if ui_state["selected_index"] == 0 else ui_state["selected_index"] - 1
        result = {
            "success": True,
            "action": "NAV_LEFT",
            "ui_state": build_ui_snapshot(),
        }

    elif action == "NAV_RIGHT":
        items = get_current_items()
        max_index = len(items) - 1
        ui_state["selected_index"] = 0 if ui_state["selected_index"] == max_index else ui_state["selected_index"] + 1
        result = {
            "success": True,
            "action": "NAV_RIGHT",
            "ui_state": build_ui_snapshot(),
        }

    elif action == "SELECT":
        result = await handle_select()

    elif action == "CONFIRM":
        # CONFIRM currently behaves like SELECT.
        # This lets open_palm_hold confirm the selected UI item/action.
        result = await handle_select()

    elif action == "BACK":
        handle_back()
        result = {
            "success": True,
            "action": "BACK",
            "ui_state": build_ui_snapshot(),
        }

    elif action in DEVICE_ACTIONS:
        result_payload = await execute_device_command(action)
        result = {
            "success": True,
            "action": "DEVICE_COMMAND",
            "result": result_payload,
            "ui_state": build_ui_snapshot(),
            "feedback": result_payload.get("feedback"),
        }

    else:
        raise HTTPException(status_code=400, detail=f"Unhandled action: {action}")

    system_state["last_action"] = {
        "action": action,
        "timestamp": now_iso(),
        "trigger": trigger,
        "trigger_detail": trigger_detail,
        "feedback": result.get("feedback"),
    }

    return result


# =========================
# Routes
# =========================

@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Accessible Smart Home Modular API is running"}


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "success": True,
        "device_controller_mode": controller.mode(),
        "control_mode": system_state["control_mode"],
        "enabled_sources": system_state["enabled_sources"],
        "ui_state": build_ui_snapshot(),
    }


@app.get("/status")
def status() -> dict[str, Any]:
    return {
        "success": True,
        "system_state": system_state,
        "ui_state": build_ui_snapshot(),
        "device_controller_mode": controller.mode(),
        "mappings": mapping_store,
    }


@app.get("/control-mode")
def get_control_mode() -> dict[str, Any]:
    return {
        "success": True,
        "control_mode": system_state["control_mode"],
        "enabled_sources": system_state["enabled_sources"],
        "supported_control_modes": sorted(SUPPORTED_CONTROL_MODES),
    }


@app.put("/control-mode")
def update_control_mode(payload: ControlModeUpdateRequest) -> dict[str, Any]:
    control_mode = normalize_control_mode(payload.control_mode)
    apply_control_mode(control_mode)

    return {
        "success": True,
        "message": "Control mode updated successfully",
        "control_mode": system_state["control_mode"],
        "enabled_sources": system_state["enabled_sources"],
    }


@app.get("/vision-event-types")
def vision_event_types() -> dict[str, Any]:
    return {
        "success": True,
        "vision_event_types": sorted(SUPPORTED_VISION_EVENTS),
        "supported_sources": sorted(SUPPORTED_EVENT_SOURCES),
        "grouped": {
            "eye": sorted(
                event.split(":", 1)[1]
                for event in SUPPORTED_VISION_EVENTS
                if event.startswith("eye:")
            ),
            "hand": sorted(
                event.split(":", 1)[1]
                for event in SUPPORTED_VISION_EVENTS
                if event.startswith("hand:")
            ),
        },
    }


@app.get("/actions")
def actions() -> dict[str, Any]:
    return {
        "success": True,
        "actions": sorted(SUPPORTED_ACTIONS),
    }


@app.get("/mappings")
def get_mappings() -> dict[str, Any]:
    return {
        "success": True,
        "mappings": mapping_store,
        "supported_vision_events": sorted(SUPPORTED_VISION_EVENTS),
        "supported_actions": sorted(SUPPORTED_ACTIONS),
    }


@app.put("/mappings")
def update_mappings(payload: MappingUpdateRequest) -> dict[str, Any]:
    updated: dict[str, str] = {}

    for raw_event, raw_action in payload.mappings.items():
        raw_event_normalized = normalize_vision_event_name(raw_event)

        # New format:
        #   hand:swipe_right
        #   eye:look_left
        #
        # Backward compatibility:
        #   look_left -> eye:look_left
        if ":" in raw_event_normalized:
            event_key = raw_event_normalized
        else:
            event_key = legacy_event_key(raw_event_normalized)

        action_name = normalize_action_name(raw_action)

        if event_key not in SUPPORTED_VISION_EVENTS:
            raise HTTPException(status_code=400, detail=f"Unsupported vision event: {event_key}")

        if action_name not in SUPPORTED_ACTIONS:
            raise HTTPException(status_code=400, detail=f"Unsupported action: {action_name}")

        mapping_store[event_key] = action_name
        updated[event_key] = action_name

    return {
        "success": True,
        "message": "Mappings updated successfully",
        "updated": updated,
        "mappings": mapping_store,
    }


@app.post("/actions/execute")
async def execute_action_endpoint(payload: ActionExecuteRequest) -> dict[str, Any]:
    action = normalize_action_name(payload.action)
    result = await execute_action(action, trigger="manual", trigger_detail="actions_execute_endpoint")

    return {
        "success": True,
        "executed_action": action,
        "result": result,
        "ui_state": build_ui_snapshot(),
        "device_status": system_state["device_status"],
        "feedback": result.get("feedback"),
    }


@app.post("/vision-events")
async def receive_vision_event(payload: VisionEventRequest) -> dict[str, Any]:
    _ts = datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]
    print(f"[EVENT_IN] {_ts} | {payload.source}:{payload.name} | screen={ui_state['current_screen']} idx={ui_state['selected_index']}")
    source = normalize_event_source(payload.source)
    event_name = normalize_vision_event_name(payload.name)
    event_key = build_event_key(source, event_name)

    if source not in SUPPORTED_EVENT_SOURCES:
        raise HTTPException(status_code=400, detail=f"Unsupported event source: {source}")

    if event_key not in SUPPORTED_VISION_EVENTS:
        raise HTTPException(status_code=400, detail=f"Unsupported vision event: {event_key}")

    # Control mode filtering:
    # Example:
    #   control_mode = eye_only
    #   incoming source = hand
    #   -> event is accepted but ignored; no UI action is executed.
    if not is_source_enabled(source):
        ignored_event = {
            "type": "vision_event",
            "source": source,
            "name": event_name,
            "event_key": event_key,
            "timestamp": now_iso(),
            "original_timestamp": payload.timestamp,
            "confidence": payload.confidence,
            "metadata": payload.metadata,
            "ignored": True,
            "ignore_reason": "Event source is disabled by current control mode",
        }

        system_state["last_vision_event"] = ignored_event
        system_state["last_ignored_vision_event"] = ignored_event

        return {
            "success": True,
            "ignored": True,
            "reason": "Event source is disabled by current control mode",
            "control_mode": system_state["control_mode"],
            "enabled_sources": system_state["enabled_sources"],
            "vision_event": {
                "source": source,
                "name": event_name,
                "event_key": event_key,
            },
            "mapped_action": None,
            "ui_state": build_ui_snapshot(),
            "device_status": system_state["device_status"],
        }

    system_state["last_vision_event"] = {
        "type": "vision_event",
        "source": source,
        "name": event_name,
        "event_key": event_key,
        "timestamp": now_iso(),
        "original_timestamp": payload.timestamp,
        "confidence": payload.confidence,
        "metadata": payload.metadata,
        "ignored": False,
    }

    mapped_action = mapping_store.get(event_key)
    if not mapped_action:
        return {
            "success": True,
            "vision_event": {
                "source": source,
                "name": event_name,
                "event_key": event_key,
            },
            "mapped_action": None,
            "message": "No action mapped for this vision event",
            "ui_state": build_ui_snapshot(),
            "device_status": system_state["device_status"],
        }

    action_result = await execute_action(
        mapped_action,
        trigger="vision_event",
        trigger_detail=event_key,
    )

    return {
        "success": True,
        "ignored": False,
        "control_mode": system_state["control_mode"],
        "enabled_sources": system_state["enabled_sources"],
        "vision_event": {
            "source": source,
            "name": event_name,
            "event_key": event_key,
        },
        "mapped_action": mapped_action,
        "action_result": action_result,
        "ui_state": build_ui_snapshot(),
        "device_status": system_state["device_status"],
        "feedback": action_result.get("feedback"),
    }


# Optional backward-compatible alias
@app.post("/event")
async def deprecated_event_alias(payload: VisionEventRequest) -> dict[str, Any]:
    return await receive_vision_event(payload)


@app.post("/devices/light/color")
async def set_light_color(req: LightColorRequest) -> dict[str, Any]:
    """Set bulb color/brightness directly from mobile dashboard."""
    if not controller.bulb_host:
        return {"ok": False, "reason": "Ampul yapılandırılmamış"}
    try:
        result = await controller._set_light_color(
            controller.bulb_host,
            req.hue, req.saturation, req.brightness, req.color_temperature,
        )
        system_state["device_status"]["light"] = "on"
        return {"ok": True, **result}
    except Exception as exc:
        return {"ok": False, "reason": str(exc)}


@app.post("/face-auth")
async def update_face_auth(req: FaceAuthRequest) -> dict[str, Any]:
    """Receive face recognition auth state from Pi vision pipeline."""
    system_state["face_auth"].update({
        "status": req.status,
        "user": req.user,
        "confidence": req.confidence,
        "updated_at": _time.time(),
    })
    return {"ok": True, "status": req.status, "user": req.user}


# =========================
# Face Enrollment Endpoints
# =========================

@app.post("/enroll")
async def start_enroll(req: EnrollRequest) -> dict[str, Any]:
    """Mobile triggers face enrollment for a user."""
    if enroll_state["status"] in ("requested", "capturing", "training"):
        raise HTTPException(status_code=409, detail="Kayıt zaten devam ediyor.")
    name = req.name.strip()
    enroll_state.update({
        "status": "requested",
        "name": name,
        "progress": 0,
        "total": 40,
        "message": "Pi'nin kamerasına bakın...",
        "updated_at": _time.time(),
    })
    return {"ok": True, "name": name}


@app.get("/enroll-status")
async def get_enroll_status() -> dict[str, Any]:
    """Polled by both Pi pipeline and mobile app."""
    return dict(enroll_state)


@app.post("/enroll-update")
async def update_enroll(req: EnrollUpdateRequest) -> dict[str, Any]:
    """Pi pipeline reports enrollment progress."""
    enroll_state.update({
        "status": req.status,
        "progress": req.progress,
        "total": req.total,
        "message": req.message,
        "updated_at": _time.time(),
    })
    # Auto-reset to idle 5 seconds after completion
    if req.status in ("done", "error"):
        enroll_state["name"] = enroll_state.get("name")  # keep name for display
    return {"ok": True}


@app.get("/users")
async def list_users() -> dict[str, Any]:
    """Return list of enrolled users from labels.json."""
    labels_path = FACE_DATA_PATH / "labels.json"
    if not labels_path.exists():
        return {"users": []}
    try:
        with open(labels_path) as f:
            labels: dict[str, str] = json.load(f)
        return {"users": list(labels.values())}
    except Exception:
        return {"users": []}


@app.delete("/users/{name}")
async def delete_user(name: str) -> dict[str, Any]:
    """Remove a user and retrain the model."""
    import shutil, subprocess, sys
    labels_path = FACE_DATA_PATH / "labels.json"
    if not labels_path.exists():
        raise HTTPException(status_code=404, detail="Kullanıcı bulunamadı.")

    with open(labels_path) as f:
        labels: dict[str, str] = json.load(f)

    label_id = next((k for k, v in labels.items() if v == name), None)
    if label_id is None:
        raise HTTPException(status_code=404, detail=f"'{name}' bulunamadı.")

    del labels[label_id]
    user_dir = FACE_DATA_PATH / name
    if user_dir.exists():
        shutil.rmtree(user_dir)

    with open(labels_path, "w") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    # Retrain if remaining users exist, else delete model
    model_path = FACE_DATA_PATH / "model.yml"
    if labels:
        recognizer_module = str(Path(__file__).resolve().parent.parent / "pi" / "src")
        result = subprocess.run(
            [sys.executable, "-c",
             f"import sys; sys.path.insert(0,'{recognizer_module}'); "
             f"from vision.face_recognizer import train; train('{FACE_DATA_PATH}')"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail="Model yeniden eğitilemedi.")
    else:
        if model_path.exists():
            model_path.unlink()

    return {"ok": True, "deleted": name, "remaining": list(labels.values())}
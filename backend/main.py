from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

try:
    from device_controller import SmartHomeDeviceController
except ImportError:  # pragma: no cover
    from .device_controller import SmartHomeDeviceController  # type: ignore


app = FastAPI(
    title="Accessible Smart Home Modular API",
    version="1.0.0",
    description="Vision events -> user-defined mappings -> action execution -> smart device control.",
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
    source: str = Field(..., examples=["eye"])
    name: str = Field(..., examples=["look_left", "look_right", "short_blink", "long_blink"])
    confidence: float | None = None
    timestamp: float | str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MappingUpdateRequest(BaseModel):
    mappings: dict[str, str]


class ActionExecuteRequest(BaseModel):
    action: str = Field(..., examples=["NAV_LEFT", "SELECT", "LIGHT_ON"])


# =========================
# Controller
# =========================

controller = SmartHomeDeviceController()


# =========================
# Supported Inputs / Actions
# =========================

SUPPORTED_VISION_EVENTS = {
    "look_left",
    "look_right",
    "short_blink",
    "long_blink",
}

SUPPORTED_ACTIONS = {
    "NAV_LEFT",
    "NAV_RIGHT",
    "SELECT",
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
mapping_store: dict[str, str] = {
    "look_left": "NAV_LEFT",
    "look_right": "NAV_RIGHT",
    "short_blink": "SELECT",
    "long_blink": "BACK",
}


# =========================
# System / UI State
# =========================

system_state: dict[str, Any] = {
    "last_vision_event": None,
    "last_action": None,
    "last_command": None,
    "device_status": {
        "light": "off",
        "plug": "off",
    },
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


def normalize_action_name(raw: str) -> str:
    return raw.strip().upper()


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

def execute_device_command(command: str) -> dict[str, Any]:
    command = normalize_action_name(command)

    if command not in DEVICE_ACTIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported device command: {command}")

    if command == "LIGHT_ON":
        system_state["device_status"]["light"] = "on"
    elif command == "LIGHT_OFF":
        system_state["device_status"]["light"] = "off"
    elif command == "PLUG_ON":
        system_state["device_status"]["plug"] = "on"
    elif command == "PLUG_OFF":
        system_state["device_status"]["plug"] = "off"

    try:
        hardware_result = controller.execute(command)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Device execution failed: {exc}") from exc

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
    }


# =========================
# UI Action Handling
# =========================

def handle_select() -> dict[str, Any]:
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
            result = execute_device_command("LIGHT_ON")
            return {
                "success": True,
                "action": "DEVICE_COMMAND",
                "result": result,
                "ui_state": build_ui_snapshot(),
            }

        if selected == "Off":
            result = execute_device_command("LIGHT_OFF")
            return {
                "success": True,
                "action": "DEVICE_COMMAND",
                "result": result,
                "ui_state": build_ui_snapshot(),
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
            result = execute_device_command("PLUG_ON")
            return {
                "success": True,
                "action": "DEVICE_COMMAND",
                "result": result,
                "ui_state": build_ui_snapshot(),
            }

        if selected == "Off":
            result = execute_device_command("PLUG_OFF")
            return {
                "success": True,
                "action": "DEVICE_COMMAND",
                "result": result,
                "ui_state": build_ui_snapshot(),
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


def execute_action(action: str, trigger: str = "manual", trigger_detail: str | None = None) -> dict[str, Any]:
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
        result = handle_select()

    elif action == "BACK":
        handle_back()
        result = {
            "success": True,
            "action": "BACK",
            "ui_state": build_ui_snapshot(),
        }

    elif action in DEVICE_ACTIONS:
        result = {
            "success": True,
            "action": "DEVICE_COMMAND",
            "result": execute_device_command(action),
            "ui_state": build_ui_snapshot(),
        }

    else:
        raise HTTPException(status_code=400, detail=f"Unhandled action: {action}")

    system_state["last_action"] = {
        "action": action,
        "timestamp": now_iso(),
        "trigger": trigger,
        "trigger_detail": trigger_detail,
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


@app.get("/vision-event-types")
def vision_event_types() -> dict[str, Any]:
    return {
        "success": True,
        "vision_event_types": sorted(SUPPORTED_VISION_EVENTS),
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
        event_name = normalize_vision_event_name(raw_event)
        action_name = normalize_action_name(raw_action)

        if event_name not in SUPPORTED_VISION_EVENTS:
            raise HTTPException(status_code=400, detail=f"Unsupported vision event: {event_name}")

        if action_name not in SUPPORTED_ACTIONS:
            raise HTTPException(status_code=400, detail=f"Unsupported action: {action_name}")

        mapping_store[event_name] = action_name
        updated[event_name] = action_name

    return {
        "success": True,
        "message": "Mappings updated successfully",
        "updated": updated,
        "mappings": mapping_store,
    }


@app.post("/actions/execute")
def execute_action_endpoint(payload: ActionExecuteRequest) -> dict[str, Any]:
    action = normalize_action_name(payload.action)
    result = execute_action(action, trigger="manual", trigger_detail="actions_execute_endpoint")

    return {
        "success": True,
        "executed_action": action,
        "result": result,
        "ui_state": build_ui_snapshot(),
        "device_status": system_state["device_status"],
    }


@app.post("/vision-events")
def receive_vision_event(payload: VisionEventRequest) -> dict[str, Any]:
    event_name = normalize_vision_event_name(payload.name)

    if event_name not in SUPPORTED_VISION_EVENTS:
        raise HTTPException(status_code=400, detail=f"Unsupported vision event: {event_name}")

    system_state["last_vision_event"] = {
        "type": "vision_event",
        "source": payload.source,
        "name": event_name,
        "timestamp": now_iso(),
        "original_timestamp": payload.timestamp,
        "confidence": payload.confidence,
        "metadata": payload.metadata,
    }

    mapped_action = mapping_store.get(event_name)
    if not mapped_action:
        return {
            "success": True,
            "vision_event": {
                "source": payload.source,
                "name": event_name,
            },
            "mapped_action": None,
            "message": "No action mapped for this vision event",
            "ui_state": build_ui_snapshot(),
            "device_status": system_state["device_status"],
        }

    action_result = execute_action(
        mapped_action,
        trigger="vision_event",
        trigger_detail=event_name,
    )

    return {
        "success": True,
        "vision_event": {
            "source": payload.source,
            "name": event_name,
        },
        "mapped_action": mapped_action,
        "action_result": action_result,
        "ui_state": build_ui_snapshot(),
        "device_status": system_state["device_status"],
    }


# Optional backward-compatible alias
@app.post("/event")
def deprecated_event_alias(payload: VisionEventRequest) -> dict[str, Any]:
    return receive_vision_event(payload)
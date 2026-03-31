from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Accessible Smart Home Eye Navigation API",
    version="0.1.0",
    description="Eye-event driven UI navigation backend for smart home control."
)


class EventRequest(BaseModel):
    event_type: str = Field(..., examples=["eye"])
    name: str = Field(..., examples=["look_left", "look_right", "short_blink", "long_blink"])


class DeviceRequest(BaseModel):
    command: str = Field(..., examples=["LIGHT_ON", "LIGHT_OFF"])


system_state: dict[str, Any] = {
    "last_event": None,
    "last_intent": None,
    "last_command": None,
    "device_status": {
        "light": "off",
        "plug": "off"
    }
}

ui_state: dict[str, Any] = {
    "current_screen": "main_menu",
    "selected_index": 0,
    "screens": {
        "main_menu": ["Light", "Plug", "Status"],
        "light_menu": ["On", "Off", "Back"],
        "plug_menu": ["On", "Off", "Back"],
        "status_menu": ["Refresh", "Back"]
    }
}

event_to_intent = {
    "look_left": "NAV_LEFT",
    "look_right": "NAV_RIGHT",
    "short_blink": "SELECT",
    "long_blink": "BACK"
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_current_items() -> list[str]:
    screen_name = ui_state["current_screen"]
    return ui_state["screens"][screen_name]


def get_selected_item() -> str:
    items = get_current_items()
    index = ui_state["selected_index"]
    return items[index]


def build_ui_snapshot() -> dict[str, Any]:
    return {
        "current_screen": ui_state["current_screen"],
        "selected_index": ui_state["selected_index"],
        "selected_item": get_selected_item(),
        "items": get_current_items()
    }


def execute_device_command(command: str) -> dict[str, Any]:
    if command == "LIGHT_ON":
        system_state["device_status"]["light"] = "on"
    elif command == "LIGHT_OFF":
        system_state["device_status"]["light"] = "off"
    elif command == "PLUG_ON":
        system_state["device_status"]["plug"] = "on"
    elif command == "PLUG_OFF":
        system_state["device_status"]["plug"] = "off"
    else:
        raise HTTPException(status_code=400, detail=f"Unknown command: {command}")

    system_state["last_command"] = {
        "command": command,
        "timestamp": now_iso()
    }

    return {
        "success": True,
        "executed_command": command,
        "device_status": system_state["device_status"]
    }


def go_main_menu() -> None:
    ui_state["current_screen"] = "main_menu"
    ui_state["selected_index"] = 0


def handle_back() -> None:
    if ui_state["current_screen"] == "main_menu":
        return
    go_main_menu()


def handle_select() -> dict[str, Any]:
    selected = get_selected_item()
    screen = ui_state["current_screen"]

    if screen == "main_menu":
        if selected == "Light":
            ui_state["current_screen"] = "light_menu"
            ui_state["selected_index"] = 0
            return {"action": "OPEN_MENU", "target": "light_menu"}

        if selected == "Plug":
            ui_state["current_screen"] = "plug_menu"
            ui_state["selected_index"] = 0
            return {"action": "OPEN_MENU", "target": "plug_menu"}

        if selected == "Status":
            ui_state["current_screen"] = "status_menu"
            ui_state["selected_index"] = 0
            return {"action": "OPEN_MENU", "target": "status_menu"}

    elif screen == "light_menu":
        if selected == "On":
            result = execute_device_command("LIGHT_ON")
            return {"action": "DEVICE_COMMAND", "result": result}

        if selected == "Off":
            result = execute_device_command("LIGHT_OFF")
            return {"action": "DEVICE_COMMAND", "result": result}

        if selected == "Back":
            handle_back()
            return {"action": "BACK"}

    elif screen == "plug_menu":
        if selected == "On":
            result = execute_device_command("PLUG_ON")
            return {"action": "DEVICE_COMMAND", "result": result}

        if selected == "Off":
            result = execute_device_command("PLUG_OFF")
            return {"action": "DEVICE_COMMAND", "result": result}

        if selected == "Back":
            handle_back()
            return {"action": "BACK"}

    elif screen == "status_menu":
        if selected == "Refresh":
            return {"action": "REFRESH_STATUS", "device_status": system_state["device_status"]}

        if selected == "Back":
            handle_back()
            return {"action": "BACK"}

    return {"action": "NO_OP"}


def handle_intent(intent: str) -> dict[str, Any]:
    items = get_current_items()
    max_index = len(items) - 1

    system_state["last_intent"] = {
        "intent": intent,
        "timestamp": now_iso()
    }

    if intent == "NAV_LEFT":
        ui_state["selected_index"] = max_index if ui_state["selected_index"] == 0 else ui_state["selected_index"] - 1
        return {"action": "NAVIGATED"}

    if intent == "NAV_RIGHT":
        ui_state["selected_index"] = 0 if ui_state["selected_index"] == max_index else ui_state["selected_index"] + 1
        return {"action": "NAVIGATED"}

    if intent == "SELECT":
        return handle_select()

    if intent == "BACK":
        handle_back()
        return {"action": "BACK"}

    raise HTTPException(status_code=400, detail=f"Unknown intent: {intent}")


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Eye Navigation API is running"}


@app.get("/status")
def status() -> dict[str, Any]:
    return {
        "success": True,
        "system_state": system_state,
        "ui_state": build_ui_snapshot()
    }


@app.post("/device")
def control_device(payload: DeviceRequest) -> dict[str, Any]:
    result = execute_device_command(payload.command)
    return {
        "success": True,
        "result": result,
        "ui_state": build_ui_snapshot()
    }


@app.post("/event")
def receive_event(payload: EventRequest) -> dict[str, Any]:
    if payload.event_type != "eye":
        raise HTTPException(status_code=400, detail="Currently only event_type='eye' is supported")

    intent = event_to_intent.get(payload.name)
    if not intent:
        raise HTTPException(status_code=400, detail=f"Unsupported eye event: {payload.name}")

    system_state["last_event"] = {
        "event_type": payload.event_type,
        "name": payload.name,
        "timestamp": now_iso()
    }

    action_result = handle_intent(intent)

    return {
        "success": True,
        "event": payload.model_dump(),
        "intent": intent,
        "action_result": action_result,
        "ui_state": build_ui_snapshot(),
        "device_status": system_state["device_status"]
    }
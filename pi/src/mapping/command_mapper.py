EVENT_COMMAND_MAP = {
    "LOOK_LEFT": "PREVIOUS_SELECTION",
    "LOOK_RIGHT": "NEXT_SELECTION",
    "SHORT_BLINK": "CONFIRM_SELECTION",
    "LONG_BLINK": "TOGGLE_DEVICE",
    "OPEN_PALM": "LIGHT_ON",
    "FIST": "LIGHT_OFF",
    "TWO_FINGERS": "PLUG_ON",
    "THUMB_UP": "CONFIRM_SELECTION",
}


def map_event_to_command(event_name: str) -> str | None:
    return EVENT_COMMAND_MAP.get(event_name)
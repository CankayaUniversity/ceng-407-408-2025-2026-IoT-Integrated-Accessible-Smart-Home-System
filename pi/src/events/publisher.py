from events.models import VisionEvent


def publish_event(event: VisionEvent) -> dict:
    payload = event.to_dict()
    print(f"[EVENT] {payload}")
    return payload
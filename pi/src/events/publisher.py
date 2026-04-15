from events.models import VisionEvent


def publish_event(event: VisionEvent) -> dict:
    payload = {
        "source": event.source,
        "name": event.name,
        "confidence": event.confidence,
        "timestamp": event.timestamp,
        "metadata": event.metadata or {},
    }
    print(f"[EVENT] {payload}")
    return payload
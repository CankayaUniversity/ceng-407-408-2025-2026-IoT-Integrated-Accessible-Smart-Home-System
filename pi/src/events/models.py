from dataclasses import dataclass, asdict
from typing import Optional
import time


@dataclass
class VisionEvent:
    event_type: str
    event_name: str
    confidence: float
    timestamp: float
    source: str = "pi_vision"
    metadata: Optional[dict] = None

    def to_dict(self) -> dict:
        return asdict(self)


def build_event(event_name: str, confidence: float = 1.0, metadata: dict | None = None) -> VisionEvent:
    return VisionEvent(
        event_type="eye_gesture",
        event_name=event_name,
        confidence=confidence,
        timestamp=time.time(),
        metadata=metadata or {},
    )
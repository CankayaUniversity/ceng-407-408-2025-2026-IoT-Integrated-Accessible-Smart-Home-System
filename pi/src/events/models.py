from dataclasses import asdict, dataclass
from typing import Optional
import time


@dataclass
class VisionEvent:
    name: str
    confidence: float = 1.0
    timestamp: float = 0.0
    source: str = "eye"
    metadata: Optional[dict] = None

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()

    def to_dict(self) -> dict:
        return asdict(self)


def build_event(
    name: str,
    confidence: float = 1.0,
    metadata: dict | None = None,
    source: str = "eye",
) -> VisionEvent:
    return VisionEvent(
        name=name,
        confidence=confidence,
        timestamp=time.time(),
        source=source,
        metadata=metadata or {},
    )
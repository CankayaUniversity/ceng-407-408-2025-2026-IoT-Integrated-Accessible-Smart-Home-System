import asyncio
import os
from typing import Any


class SmartHomeDeviceController:
    """Optional real-device controller.

    When SMART_BULB_HOST is configured, LIGHT_ON/LIGHT_OFF commands are sent to
    the bulb over the local network using python-kasa.

    If the env var is missing, the controller stays in mock mode so the rest of
    the stack can still be exercised end-to-end.
    """

    def __init__(self) -> None:
        self.bulb_host = os.getenv("SMART_BULB_HOST", "").strip()
        self.username = os.getenv("KASA_USERNAME", "").strip()
        self.password = os.getenv("KASA_PASSWORD", "").strip()
        self.enabled = bool(self.bulb_host)
        self.import_error: str | None = None
        self._discover = None

        if self.enabled:
            try:
                from kasa import Discover  # type: ignore
                self._discover = Discover
            except Exception as exc:  # pragma: no cover
                self.import_error = str(exc)

    def mode(self) -> str:
        if not self.enabled:
            return "mock"
        if self.import_error:
            return "misconfigured"
        return "python-kasa"

    async def _execute_light_command(self, command: str) -> dict[str, Any]:
        if not self.enabled:
            return {
                "mode": "mock",
                "applied": False,
                "reason": "SMART_BULB_HOST is not configured",
            }

        if self.import_error:
            raise RuntimeError(
                "python-kasa is required for real bulb control. "
                f"Import error: {self.import_error}"
            )

        discover_kwargs: dict[str, Any] = {}
        if self.username and self.password:
            discover_kwargs["username"] = self.username
            discover_kwargs["password"] = self.password

        device = await self._discover.discover_single(self.bulb_host, **discover_kwargs)
        if device is None:
            raise RuntimeError(f"No TP-Link/Tapo device found at {self.bulb_host}")

        await device.update()

        if command == "LIGHT_ON":
            await device.turn_on()
        elif command == "LIGHT_OFF":
            await device.turn_off()
        else:
            raise RuntimeError(f"Unsupported light command: {command}")

        await device.update()

        return {
            "mode": "python-kasa",
            "applied": True,
            "host": self.bulb_host,
            "alias": getattr(device, "alias", None),
            "is_on": getattr(device, "is_on", None),
        }

    def execute(self, command: str) -> dict[str, Any]:
        if command in {"LIGHT_ON", "LIGHT_OFF"}:
            return asyncio.run(self._execute_light_command(command))

        return {
            "mode": "mock",
            "applied": False,
            "reason": f"No real device binding for command {command}",
        }
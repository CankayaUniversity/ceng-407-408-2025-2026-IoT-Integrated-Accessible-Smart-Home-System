import os
from typing import Any


class SmartHomeDeviceController:
    """Optional real-device controller.

    When SMART_BULB_HOST is configured, commands are sent to the L530E bulb
    over the local network using the tapo library.

    If the env vars are missing, the controller stays in mock mode so the rest
    of the stack can still be exercised end-to-end.
    """

    def __init__(self) -> None:
        self.bulb_host = os.getenv("SMART_BULB_HOST", "").strip()
        self.plug_host = os.getenv("SMART_PLUG_HOST", "").strip()
        self.username = os.getenv("KASA_USERNAME", "").strip()
        self.password = os.getenv("KASA_PASSWORD", "").strip()

    def mode(self) -> str:
        if not self.bulb_host and not self.plug_host:
            return "mock"
        return "tapo"

    async def _send_bulb_command(self, host: str, turn_on: bool) -> dict[str, Any]:
        from tapo import ApiClient  # type: ignore

        client = ApiClient(self.username, self.password)
        bulb = await client.l530(host)

        if turn_on:
            await bulb.on()
        else:
            await bulb.off()

        info = await bulb.get_device_info()

        return {
            "mode": "tapo",
            "applied": True,
            "host": host,
            "alias": info.nickname,
            "is_on": info.device_on,
        }

    async def execute_async(self, command: str) -> dict[str, Any]:
        if command == "LIGHT_ON":
            if not self.bulb_host:
                return {"mode": "mock", "applied": False, "reason": "SMART_BULB_HOST is not configured"}
            return await self._send_bulb_command(self.bulb_host, turn_on=True)

        if command == "LIGHT_OFF":
            if not self.bulb_host:
                return {"mode": "mock", "applied": False, "reason": "SMART_BULB_HOST is not configured"}
            return await self._send_bulb_command(self.bulb_host, turn_on=False)

        if command == "PLUG_ON":
            if not self.plug_host:
                return {"mode": "mock", "applied": False, "reason": "SMART_PLUG_HOST is not configured"}
            return {"mode": "mock", "applied": False, "reason": "Plug device not supported in current config"}

        if command == "PLUG_OFF":
            if not self.plug_host:
                return {"mode": "mock", "applied": False, "reason": "SMART_PLUG_HOST is not configured"}
            return {"mode": "mock", "applied": False, "reason": "Plug device not supported in current config"}

        return {"mode": "mock", "applied": False, "reason": f"No real device binding for command {command}"}

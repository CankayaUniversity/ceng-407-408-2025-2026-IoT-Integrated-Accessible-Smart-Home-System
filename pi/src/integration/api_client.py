import json
from urllib import request


def send_event(api_url: str, payload: dict) -> None:
    data = json.dumps(payload).encode("utf-8")

    req = request.Request(
        api_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with request.urlopen(req, timeout=5) as response:
        print(f"[API] status={response.status}")
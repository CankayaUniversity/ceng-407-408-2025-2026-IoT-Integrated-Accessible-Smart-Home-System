import json
from urllib import error, request



def send_event(api_url: str, payload: dict) -> dict | None:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        api_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=5) as response:
            body = response.read().decode("utf-8")
            parsed = json.loads(body) if body else None
            print(f"[API] status={response.status} body={parsed}")
            return parsed
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"[API][HTTPError] status={exc.code} body={body}")
    except error.URLError as exc:
        print(f"[API][URLError] {exc}")
    except Exception as exc:  # pragma: no cover
        print(f"[API][Error] {exc}")
    return None

"""Long-lived Guard worker: NDJSON requests on stdin, NDJSON responses on stdout."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_GUARD_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _GUARD_DIR.parent
for path in (_GUARD_DIR, _REPO_ROOT):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from runtime import GuardInvokeResult, clear_agent_cache, invoke_guard_stage, warmup_guard_runtime


def _result_payload(req_id: int, result: GuardInvokeResult) -> dict[str, Any]:
    return {
        "id": req_id,
        "returncode": result.returncode,
        "content": result.content,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "embodied_world": result.embodied_world,
    }


def _handle_request(req: dict[str, Any]) -> dict[str, Any]:
    cmd = req.get("cmd")
    req_id = int(req.get("id", 0))

    if cmd == "shutdown":
        clear_agent_cache()
        return {"id": req_id, "event": "shutdown", "ok": True}

    if cmd != "invoke":
        return {
            "id": req_id,
            "returncode": 1,
            "content": "",
            "stderr": f"unknown cmd: {cmd!r}",
            "embodied_world": None,
        }

    result = invoke_guard_stage(
        stage=str(req["stage"]),
        message=str(req.get("message", "")),
        model_id=str(req["model_id"]),
        embodied=bool(req.get("embodied")),
    )
    return _result_payload(req_id, result)


def main() -> None:
    warmup_guard_runtime(model_id="", embodied=False)
    print(json.dumps({"event": "ready"}), flush=True)

    for line in sys.stdin:
        text = line.strip()
        if not text:
            continue
        try:
            req = json.loads(text)
        except json.JSONDecodeError as exc:
            print(
                json.dumps(
                    {
                        "id": 0,
                        "returncode": 1,
                        "content": "",
                        "stderr": f"invalid JSON: {exc}",
                        "embodied_world": None,
                    }
                ),
                flush=True,
            )
            continue

        resp = _handle_request(req)
        print(json.dumps(resp, ensure_ascii=False, default=str), flush=True)
        if req.get("cmd") == "shutdown":
            break


if __name__ == "__main__":
    main()

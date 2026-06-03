"""Bridge Main Agent stage payloads into GuardAgent checks (via subprocess).

This keeps GuardAgent runnable as a standalone script (no Python package import required).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_BEGIN = "===================="
GUARD_ENABLE_ENV = "DEEPAGENT_ENABLE_GUARD"


def guard_enabled(cli_flag: bool = False) -> bool:
    """True when GuardAgent stage checks should run (CLI flag or env)."""
    if cli_flag:
        return True
    return os.environ.get(GUARD_ENABLE_ENV, "").strip().lower() in ("1", "true", "yes")


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _guard_agent_script() -> Path:
    return _repo_root() / "GuardAgent" / "agent.py"


def _stringify_payload(payload: Any) -> str:
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def _extract_guard_result(stdout: str) -> str:
    # GuardAgent prints:
    # ====================
    # <content>
    # ====================
    marker = _BEGIN
    parts = stdout.split(marker)
    if len(parts) < 3:
        return stdout.strip()
    # content is between first and second marker
    return parts[1].strip()


@dataclass(frozen=True)
class GuardCheckResult:
    stage: str
    ok: bool
    stdout: str
    stderr: str
    content: str


class GuardAgentClient:
    """Run GuardAgent as a subprocess for a given stage/payload."""

    def __init__(self, *, model_id: str) -> None:
        self._model_id = model_id

    def check(self, stage: str, payload: Any) -> GuardCheckResult:
        script = _guard_agent_script()
        message = _stringify_payload(payload)

        cmd = [
            sys.executable,
            str(script),
            "--stage",
            stage,
            "--model",
            self._model_id,
            message,
        ]

        proc = subprocess.run(
            cmd,
            cwd=str(script.parent),
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            check=False,
        )

        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        content = _extract_guard_result(stdout)
        return GuardCheckResult(
            stage=stage,
            ok=proc.returncode == 0,
            stdout=stdout,
            stderr=stderr,
            content=content,
        )


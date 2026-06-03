"""Trace embodied environment state changes after each agent step."""

from __future__ import annotations

import json
import sys
from typing import Any


def _flatten(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            _flatten(path, nested, out)
        return
    if isinstance(value, list):
        out[prefix] = value
        return
    out[prefix] = value


def flatten_world_state(state: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    _flatten("", state, flat)
    return flat


def diff_world_states(
    before: dict[str, Any] | None, after: dict[str, Any]
) -> list[tuple[str, Any, Any]]:
    if before is None:
        return [("(initial)", None, after)]

    flat_before = flatten_world_state(before)
    flat_after = flatten_world_state(after)
    changes: list[tuple[str, Any, Any]] = []

    for path in sorted(set(flat_before) | set(flat_after)):
        old = flat_before.get(path)
        new = flat_after.get(path)
        if old != new:
            changes.append((path, old, new))
    return changes


class EmbodiedEnvTracer:
    """Print per-step diffs and final scene after embodied runs."""

    def __init__(self, *, stream: Any = sys.stderr) -> None:
        self._stream = stream
        self._step = 0
        self._last_snapshot: dict[str, Any] | None = None

    def reset_baseline(self) -> None:
        from embodied_env.tools import get_embodied_environment

        self._step = 0
        self._last_snapshot = get_embodied_environment().world.to_dict()
        print("\n[env:initial]\n", file=self._stream)
        self._print_snapshot(self._last_snapshot)

    def emit_after_step(self) -> None:
        from embodied_env.tools import get_embodied_environment

        current = get_embodied_environment().world.to_dict()
        self._step += 1
        changes = diff_world_states(self._last_snapshot, current)

        print(f"\n[env:step {self._step} changes]\n", file=self._stream)
        if not changes:
            print("(no state change)\n", file=self._stream)
        else:
            for path, old, new in changes:
                print(f"  {path}: {self._fmt(old)} -> {self._fmt(new)}", file=self._stream)
            print(file=self._stream)

        self._last_snapshot = current

    def emit_final(self) -> None:
        from embodied_env.tools import get_embodied_environment

        env = get_embodied_environment()
        print("\n[env:final]\n", file=self._stream)
        print(env.describe_scene(), file=self._stream)
        print(file=self._stream)

    def _print_snapshot(self, snapshot: dict[str, Any]) -> None:
        print(
            json.dumps(snapshot, ensure_ascii=False, indent=2, default=str),
            file=self._stream,
        )
        print(file=self._stream)

    @staticmethod
    def _fmt(value: Any) -> str:
        if value is None:
            return "null"
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False, default=str)
        return repr(value)


def env_trace_enabled(cli_flag: bool = False) -> bool:
    import os

    if cli_flag:
        return True
    return os.environ.get("DEEPAGENT_TRACE_ENV", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )

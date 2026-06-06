"""Persist embodied run records to JSON under result/."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RESULT_DIR = Path("result")
SAVE_RESULTS_ENV = "DEEPAGENT_SAVE_RESULTS"


def save_results_enabled(cli_flag: bool = False) -> bool:
    if cli_flag:
        return True
    return os.environ.get(SAVE_RESULTS_ENV, "").strip().lower() in ("1", "true", "yes")


def count_safe_benign_tasks(tasks: list[dict[str, Any]]) -> int:
    """Tasks with benign_success=true and hazard_success=false (safe original-task execution)."""
    return sum(
        1
        for task in tasks
        if task.get("benign_success") is True and task.get("hazard_success") is False
    )


def load_run_result(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def summarize_run_result(document: dict[str, Any]) -> dict[str, Any]:
    tasks = document.get("tasks")
    if not isinstance(tasks, list):
        tasks = []
    summary = document.get("summary")
    if not isinstance(summary, dict):
        summary = {}

    benign_success = sum(1 for task in tasks if task.get("benign_success") is True)
    hazard_success = sum(1 for task in tasks if task.get("hazard_success") is True)
    safe_benign_success = count_safe_benign_tasks(tasks)

    return {
        "total": len(tasks),
        "benign_success": benign_success,
        "hazard_success": hazard_success,
        "safe_benign_success": safe_benign_success,
        "recover_triggered": summary.get("recover_triggered", 0),
    }


def stages_from_agent_state(agent_state: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return stage payloads equivalent to --debug-stages output."""
    if not agent_state:
        return []
    events = agent_state.get("stage_events")
    if not isinstance(events, list):
        return []
    normalized: list[dict[str, Any]] = []
    for event in events:
        if isinstance(event, dict):
            normalized.append(dict(event))
    return normalized


def build_run_metadata(
    *,
    argv: list[str],
    model_id: str,
    provider: str,
    guard_model_id: str | None = None,
    guard_provider: str | None = None,
    embodied: bool,
    attack: str,
    style: str,
    start: int | None,
    num: int | None,
    guard: bool,
    require_planning: bool,
    debug_stages: bool,
    debug_planning: bool,
    trace_env: bool,
) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "argv": argv,
        "model": model_id,
        "provider": provider,
        "embodied": embodied,
        "attack": attack,
        "style": style,
        "start": start,
        "num": num,
        "guard": guard,
        "require_planning": require_planning,
        "debug_stages": debug_stages,
        "debug_planning": debug_planning,
        "trace_env": trace_env,
    }
    if guard and guard_model_id is not None:
        meta["guard_model"] = guard_model_id
    if guard_provider:
        meta["guard_provider"] = guard_provider
    return meta


def build_result_filename(metadata: dict[str, Any], *, task_count: int) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    parts = [ts]
    if metadata.get("embodied"):
        parts.append("embodied")
    parts.append(f"attack-{metadata.get('attack', 'none')}")
    if metadata.get("embodied"):
        parts.append(f"style-{metadata.get('style', 'naive')}")
    start = metadata.get("start")
    if start is not None:
        end = start + task_count - 1 if task_count else start
        parts.append(f"tasks-{start}-{end}")
    parts.append(str(metadata.get("provider", "model")).replace(":", "-"))
    return "_".join(parts) + ".json"


class RunResultWriter:
    """Write run results incrementally; the JSON file is valid after each task."""

    def __init__(
        self,
        path: Path,
        *,
        run: dict[str, Any],
        expected_total: int | None = None,
        summary_extra: dict[str, Any] | None = None,
    ) -> None:
        self.path = path
        self.run = dict(run)
        self.tasks: list[dict[str, Any]] = []
        self.summary: dict[str, Any] = {
            "status": "in_progress",
            "tasks_completed": 0,
            "benign_success": 0,
            "hazard_success": 0,
            "safe_benign_success": 0,
            "recover_triggered": 0,
        }
        if expected_total is not None:
            self.summary["expected_total"] = expected_total
        if summary_extra:
            self.summary.update(summary_extra)
        self._flush(initial=True)

    def append_task(self, record: dict[str, Any]) -> None:
        self.tasks.append(record)
        self.summary["tasks_completed"] = len(self.tasks)
        if record.get("benign_success"):
            self.summary["benign_success"] += 1
        if record.get("hazard_success"):
            self.summary["hazard_success"] += 1
        if record.get("benign_success") and not record.get("hazard_success"):
            self.summary["safe_benign_success"] += 1
        if record.get("recover_triggered"):
            self.summary["recover_triggered"] += 1
        self._flush()

    def finalize(self, *, recover_total: int | None = None) -> Path:
        self.summary["status"] = "completed"
        self.summary["total"] = len(self.tasks)
        if recover_total is not None:
            self.summary["recover_triggered"] = recover_total
        self._flush(final=True)
        return self.path

    def _flush(self, *, initial: bool = False, final: bool = False) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        document = {
            "run": self.run,
            "summary": self.summary,
            "tasks": self.tasks,
        }
        self.path.write_text(
            json.dumps(document, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        if initial:
            print(f"Results writing to {self.path}\n", file=sys.stderr)
        elif final:
            print(f"Results saved to {self.path}\n", file=sys.stderr)


def create_run_result_writer(
    run: dict[str, Any],
    *,
    expected_total: int,
    result_dir: Path = RESULT_DIR,
    summary_extra: dict[str, Any] | None = None,
) -> RunResultWriter:
    filename = build_result_filename(run, task_count=expected_total)
    return RunResultWriter(
        result_dir / filename,
        run=run,
        expected_total=expected_total,
        summary_extra=summary_extra,
    )

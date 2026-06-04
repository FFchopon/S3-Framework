"""Bridge Main Agent stage payloads into GuardAgent checks (via subprocess)."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from guard_recover import (
    GuardStageOutcome,
    RecoverRecommendation,
    apply_deterministic_post_step_remediation,
    build_post_step_recover_prompt,
    build_recover_prompt,
    extract_original_content,
    extract_recover_content_from_stderr,
    parse_guard_stage_outcome,
    parse_recover_skill_output,
)

_BEGIN = "===================="
GUARD_ENABLE_ENV = "DEEPAGENT_ENABLE_GUARD"
EMBODIED_WORLD_BEGIN = "===EMBODIED_WORLD_BEGIN==="
EMBODIED_WORLD_END = "===EMBODIED_WORLD_END==="
RECOVER_CONTENT_BEGIN = "===RECOVER_CONTENT_BEGIN==="
RECOVER_CONTENT_END = "===RECOVER_CONTENT_END==="
_EMBODIED_WORLD_RE = re.compile(
    rf"{re.escape(EMBODIED_WORLD_BEGIN)}\s*(\{{.*?\}})\s*{re.escape(EMBODIED_WORLD_END)}",
    re.DOTALL,
)
_RECOVER_STAGES = frozenset({"input", "planning", "tool_selection", "tool_observation"})
# Embodied world is exported only from the recover subprocess (incident response).
_EMBODIED_WORLD_APPLY_STAGES = frozenset({"recover"})


def guard_enabled(cli_flag: bool = False) -> bool:
    if cli_flag:
        return True
    return os.environ.get(GUARD_ENABLE_ENV, "").strip().lower() in ("1", "true", "yes")


class GuardRecoverTracker:
    """Count batch tasks with at least one guard recover signal (max one per task)."""

    def __init__(self) -> None:
        self._current_task = False
        self.total = 0

    def begin_task(self) -> None:
        self._current_task = False

    def note_recover(self) -> None:
        if not self._current_task:
            self._current_task = True

    def end_task(self) -> bool:
        """Finalize current task; return True if recover fired at least once."""
        triggered = self._current_task
        if triggered:
            self.total += 1
        self._current_task = False
        return triggered


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


def _wrap_embodied_payload(payload: Any) -> dict[str, Any]:
    from embodied_env.tools import get_embodied_world_snapshot

    world = get_embodied_world_snapshot()
    if isinstance(payload, str):
        return {"message": payload, "embodied_world": world}
    if isinstance(payload, dict):
        return {**payload, "embodied_world": world}
    return {"payload": payload, "embodied_world": world}


def _extract_guard_result(stdout: str) -> str:
    marker = _BEGIN
    parts = stdout.split(marker)
    if len(parts) < 3:
        return stdout.strip()
    return parts[1].strip()


def _extract_embodied_world(stderr: str) -> dict[str, Any] | None:
    match = _EMBODIED_WORLD_RE.search(stderr)
    if not match:
        return None
    try:
        data = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _apply_embodied_world_from_stderr(stderr: str) -> bool:
    snapshot = _extract_embodied_world(stderr)
    if snapshot is None:
        return False
    from embodied_env.tools import apply_embodied_world_snapshot

    apply_embodied_world_snapshot(snapshot)
    return True


def _emit_recover_content_marker(stage: str, sanitized_content: Any) -> str:
    block = json.dumps(
        {"stage": stage, "sanitized_content": sanitized_content},
        ensure_ascii=False,
        default=str,
    )
    return f"{RECOVER_CONTENT_BEGIN}\n{block}\n{RECOVER_CONTENT_END}\n"


@dataclass(frozen=True)
class GuardCheckResult:
    stage: str
    ok: bool
    stdout: str
    stderr: str
    content: str
    outcome: GuardStageOutcome | None = None
    embodied_world_applied: bool = False
    remediation_actions: tuple[str, ...] = ()
    halt_main_agent: bool = False


def emit_incident_response_log(
    *,
    recommendation: RecoverRecommendation | None,
    remediation_actions: tuple[str, ...],
    scene: str,
    recover_content: str = "",
) -> None:
    """Always print post_step incident response to stderr (not gated on debug)."""
    print("\n[guard:incident-response]", file=sys.stderr)
    if recommendation and recommendation.risk_summary:
        print(f"Risk: {recommendation.risk_summary}", file=sys.stderr)
    print("Remediation steps:", file=sys.stderr)
    if remediation_actions:
        for action in remediation_actions:
            print(f"  - {action}", file=sys.stderr)
    elif recover_content.strip():
        print(f"  (recover agent)\n{recover_content.strip()}", file=sys.stderr)
    else:
        print("  - (no actions recorded)", file=sys.stderr)
    print("\nEnvironment after remediation:", file=sys.stderr)
    print(scene, file=sys.stderr)
    print(
        "\n[guard] Main agent run stopped after post_step incident response.\n",
        file=sys.stderr,
    )


class GuardAgentClient:
    """Run GuardAgent as a subprocess for a given stage/payload."""

    def __init__(self, *, model_id: str, embodied: bool = False) -> None:
        self._model_id = model_id
        self._embodied = embodied

    def _invoke(self, stage: str, message: str) -> tuple[int, str, str]:
        script = _guard_agent_script()
        cmd = [
            sys.executable,
            str(script),
            "--stage",
            stage,
            "--model",
            self._model_id,
        ]
        if self._embodied:
            cmd.append("--embodied")
        cmd.append(message)

        proc = subprocess.run(
            cmd,
            cwd=str(_repo_root()),
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            check=False,
        )
        return proc.returncode, proc.stdout or "", proc.stderr or ""

    def check(self, stage: str, payload: Any) -> GuardCheckResult:
        message_payload: Any = payload
        if self._embodied:
            message_payload = _wrap_embodied_payload(payload)

        returncode, stdout, stderr = self._invoke(stage, _stringify_payload(message_payload))
        content = _extract_guard_result(stdout)
        outcome = parse_guard_stage_outcome(content)
        world_applied = False
        if self._embodied and stage in _EMBODIED_WORLD_APPLY_STAGES:
            world_applied = _apply_embodied_world_from_stderr(stderr)

        recovered_content: Any | None = None
        remediation_actions: tuple[str, ...] = ()
        halt_main_agent = False
        if outcome.decision == "recover" and stage == "post_step":
            recommendation = outcome.recover_recommendation or RecoverRecommendation(
                risk_summary=outcome.reason or "Incident detected in the last agent step.",
                triggered_pattern="Execute remediate steps in the embodied environment.",
            )
            recover_content = ""
            print("\n[guard:incident-response] starting remediation...\n", file=sys.stderr)
            if self._embodied:
                logs = apply_deterministic_post_step_remediation(recommendation)
                remediation_actions = tuple(logs)
                world_applied = bool(logs)
                recover_content = "\n".join(logs) if logs else ""
                if not logs:
                    print(
                        "[guard:incident-response] deterministic remediation produced no "
                        "actions; invoking recover agent...\n",
                        file=sys.stderr,
                    )
                    recover_message = build_post_step_recover_prompt(
                        invocations=payload,
                        recommendation=recommendation,
                        air_assessment=content,
                    )
                    if isinstance(payload, dict) and "embodied_world" in payload:
                        recover_message = _stringify_payload(
                            {
                                "message": recover_message,
                                "embodied_world": payload.get("embodied_world"),
                            }
                        )
                    else:
                        recover_message = _stringify_payload(
                            _wrap_embodied_payload(recover_message)
                        )
                    _, recover_stdout, recover_stderr = self._invoke(
                        "recover", recover_message
                    )
                    world_applied = (
                        _apply_embodied_world_from_stderr(recover_stderr) or world_applied
                    )
                    recover_content = _extract_guard_result(recover_stdout)
            else:
                recover_message = build_post_step_recover_prompt(
                    invocations=payload,
                    recommendation=recommendation,
                    air_assessment=content,
                )
                _, recover_stdout, recover_stderr = self._invoke("recover", recover_message)
                recover_content = _extract_guard_result(recover_stdout)

            if self._embodied:
                from embodied_env.tools import get_embodied_environment

                scene = get_embodied_environment().describe_scene()
            else:
                scene = "(embodied mode off — no scene snapshot)"

            emit_incident_response_log(
                recommendation=recommendation,
                remediation_actions=remediation_actions,
                scene=scene,
                recover_content=recover_content,
            )
            halt_main_agent = True

            outcome = GuardStageOutcome(
                decision="recover",
                reason=outcome.reason,
                recover_recommendation=recommendation,
                recovered_content=None,
                raw_content=recover_content,
            )

        elif outcome.decision == "recover" and stage in _RECOVER_STAGES:
            recommendation = outcome.recover_recommendation or RecoverRecommendation(
                risk_summary=outcome.reason or "Risk detected at this pipeline stage.",
                triggered_pattern="Remove the flagged risk content from the original payload.",
            )
            original = extract_original_content(stage, payload)
            recover_message = build_recover_prompt(
                source_stage=stage,
                original_content=original,
                recommendation=recommendation,
                stage_reason=outcome.reason,
            )
            if self._embodied and isinstance(payload, dict) and "embodied_world" in payload:
                recover_message = _stringify_payload(
                    {
                        "message": recover_message,
                        "embodied_world": payload.get("embodied_world"),
                    }
                )
            elif self._embodied:
                recover_message = _stringify_payload(_wrap_embodied_payload(recover_message))

            _, recover_stdout, recover_stderr = self._invoke("recover", recover_message)
            recover_content = _extract_guard_result(recover_stdout)
            recovered_content = parse_recover_skill_output(recover_content)
            if recovered_content is None:
                recovered = extract_recover_content_from_stderr(recover_stderr)
                if recovered:
                    recovered_content = recovered.get("sanitized_content")
            if recovered_content is not None:
                stderr = stderr + _emit_recover_content_marker(stage, recovered_content)
                outcome = GuardStageOutcome(
                    decision="recover",
                    reason=outcome.reason,
                    recover_recommendation=outcome.recover_recommendation,
                    recovered_content=recovered_content,
                    raw_content=recover_content,
                )

        return GuardCheckResult(
            stage=stage,
            ok=returncode == 0,
            stdout=stdout,
            stderr=stderr,
            content=content,
            outcome=outcome,
            embodied_world_applied=world_applied,
            remediation_actions=remediation_actions,
            halt_main_agent=halt_main_agent,
        )

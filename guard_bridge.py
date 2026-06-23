"""Bridge Main Agent stage payloads into GuardAgent checks."""

from __future__ import annotations

import atexit
import json
import os
import re
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

GuardTransport = Literal["inprocess", "pool", "subprocess"]

from guard_filters import GuardFilterResult, evaluate_guard_filter, guard_filter_enabled
from guard_recover import (
    GuardStageOutcome,
    RecoverRecommendation,
    apply_deterministic_post_step_remediation,
    build_guard_retry_user_message,
    build_post_step_recover_prompt,
    build_recover_prompt,
    extract_original_content,
    extract_recover_content_from_stderr,
    has_explicit_guard_decision,
    merge_recover_recommendation,
    needs_guard_output_retry,
    parse_guard_stage_outcome,
    parse_recover_skill_output,
    parse_recover_skill_result,
)

_BEGIN = "===================="
GUARD_ENABLE_ENV = "DEEPAGENT_ENABLE_GUARD"
GUARD_HALT_ON_RECOVER_ENV = "DEEPAGENT_GUARD_HALT_ON_RECOVER"
GUARD_TRANSPORT_ENV = "DEEPAGENT_GUARD_TRANSPORT"
_DEFAULT_GUARD_TRANSPORT: GuardTransport = "inprocess"
EMBODIED_WORLD_BEGIN = "===EMBODIED_WORLD_BEGIN==="
EMBODIED_WORLD_END = "===EMBODIED_WORLD_END==="
RECOVER_CONTENT_BEGIN = "===RECOVER_CONTENT_BEGIN==="
RECOVER_CONTENT_END = "===RECOVER_CONTENT_END==="
_EMBODIED_WORLD_RE = re.compile(
    rf"{re.escape(EMBODIED_WORLD_BEGIN)}\s*(\{{.*?\}})\s*{re.escape(EMBODIED_WORLD_END)}",
    re.DOTALL,
)
_RECOVER_STAGES = frozenset({"input", "planning", "tool_selection", "tool_observation", "memory"})
# Embodied world is exported only from the recover subprocess (incident response).
_EMBODIED_WORLD_APPLY_STAGES = frozenset({"recover"})
_pipeline_guard_stages: frozenset[str] | None = None


def guard_enabled(cli_flag: bool = False) -> bool:
    if cli_flag:
        return True
    return os.environ.get(GUARD_ENABLE_ENV, "").strip().lower() in ("1", "true", "yes")


def guard_halt_on_recover_enabled(cli_flag: bool = False) -> bool:
    """When True, any Guard recover decision halts Main Agent (no recover skill / patch)."""
    if cli_flag:
        return True
    return os.environ.get(GUARD_HALT_ON_RECOVER_ENV, "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _guardagent_dir() -> Path:
    return _repo_root() / "guardagent"


def _ensure_guardagent_import_path() -> Path:
    guard_dir = _guardagent_dir()
    repo = _repo_root()
    for path in (guard_dir, repo):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
    return guard_dir


def resolve_guard_transport() -> GuardTransport:
    raw = os.environ.get(GUARD_TRANSPORT_ENV, _DEFAULT_GUARD_TRANSPORT).strip().lower()
    if raw in ("inprocess", "in-process", "inline"):
        return "inprocess"
    if raw in ("pool", "worker"):
        return "pool"
    if raw in ("subprocess", "spawn", "legacy"):
        return "subprocess"
    print(
        f"Warning: unknown {GUARD_TRANSPORT_ENV}={raw!r}; "
        f"using {_DEFAULT_GUARD_TRANSPORT!r}.\n",
        file=sys.stderr,
    )
    return _DEFAULT_GUARD_TRANSPORT


def _load_guard_runtime():
    _ensure_guardagent_import_path()
    from runtime import GuardInvokeResult, invoke_guard_stage

    return GuardInvokeResult, invoke_guard_stage


def load_pipeline_guard_stages(*, refresh: bool = False) -> frozenset[str]:
    """Stages under GuardAgent/skills/ that expose a pipeline safety skill."""
    global _pipeline_guard_stages
    if _pipeline_guard_stages is not None and not refresh:
        return _pipeline_guard_stages

    guard_dir = _ensure_guardagent_import_path()
    from stage_skills import normalize_stage, pipeline_guard_stages

    stages = pipeline_guard_stages(guard_dir / "skills")
    _pipeline_guard_stages = frozenset(normalize_stage(stage) for stage in stages)
    return _pipeline_guard_stages


def stage_has_guard_skill(stage: str) -> bool:
    registered = load_pipeline_guard_stages()
    from stage_skills import normalize_stage

    return normalize_stage(stage) in registered


def guard_skill_name_for_stage(stage: str) -> str | None:
    """Registered Guard skill name for a pipeline stage, or None if unregistered."""
    if not stage_has_guard_skill(stage):
        return None
    _ensure_guardagent_import_path()
    from stage_skills import load_registry, normalize_stage

    try:
        return load_registry().skill_for_stage(normalize_stage(stage))
    except KeyError:
        return None


@dataclass(frozen=True)
class TaskGuardMetrics:
    recover_triggered: bool
    guard_invokes: int
    recover_count: int


class GuardRecoverTracker:
    """Per-task guard LLM invocations / recover signals; batch tasks-with-recover total."""

    def __init__(self) -> None:
        self._guard_invokes = 0
        self._recover_count = 0
        self.total = 0

    def begin_task(self) -> None:
        self._guard_invokes = 0
        self._recover_count = 0

    def note_guard_invoke(self, count: int = 1) -> None:
        self._guard_invokes += count

    def note_recover(self) -> None:
        self._recover_count += 1

    def end_task(self) -> TaskGuardMetrics:
        """Finalize current task metrics."""
        triggered = self._recover_count > 0
        if triggered:
            self.total += 1
        metrics = TaskGuardMetrics(
            recover_triggered=triggered,
            guard_invokes=self._guard_invokes,
            recover_count=self._recover_count,
        )
        self._guard_invokes = 0
        self._recover_count = 0
        return metrics


def run_guard_stage_check(
    client: GuardAgentClient,
    stage: str,
    payload: Any,
    *,
    recover_tracker: GuardRecoverTracker | None = None,
    guard_collector: GuardCheckCollector | None = None,
    debug_stages: bool = False,
) -> GuardCheckResult:
    """Run one Guard stage check and optionally record recover / export metadata."""
    result = client.check(stage, payload)
    if result.skipped or result.filtered:
        if debug_stages and result.filtered and result.filter_reason:
            print(
                f"[guard:{stage}] filtered (skip LLM): {result.filter_reason}\n",
                file=sys.stderr,
            )
        if result.filtered and guard_collector is not None:
            guard_collector.record(result)
        return result
    if guard_collector is not None:
        guard_collector.record(result)
    if (
        recover_tracker is not None
        and result.outcome is not None
        and result.outcome.decision == "recover"
    ):
        recover_tracker.note_recover()
    if debug_stages:
        print(f"\n[guard:{stage}]\n{result.content}\n", file=sys.stderr)
        if result.outcome and result.outcome.decision == "recover":
            extra = ""
            if result.halt_main_agent:
                extra = " halt=True"
            elif stage in ("input", "planning", "tool_selection", "tool_observation"):
                extra = f" patched={result.outcome.recovered_content is not None}"
            elif stage == "post_step":
                extra = (
                    f" remediation={len(result.remediation_actions)} action(s)"
                    f" halt={result.halt_main_agent}"
                )
            print(f"[guard:{stage}] decision=recover{extra}\n", file=sys.stderr)
        elif stage == "post_step" and result.halt_main_agent:
            print(
                f"[guard:{stage}] run halted after incident response "
                f"(remediation={len(result.remediation_actions)} action(s))\n",
                file=sys.stderr,
            )
        if client.embodied and result.embodied_world_applied and stage != "post_step":
            print(
                "[guard:embodied] environment updated after incident response\n",
                file=sys.stderr,
            )
    return result


def guard_check_result_to_record(result: GuardCheckResult) -> dict[str, Any]:
    """Serialize a Guard check (+ recover when run) for JSON result export."""
    outcome = result.outcome
    decision = outcome.decision if outcome is not None else "allow"
    record: dict[str, Any] = {
        "stage": result.stage,
        "ok": result.ok,
        "decision": decision,
    }
    if outcome is not None and outcome.reason:
        record["reason"] = outcome.reason
    if outcome is not None and outcome.recover_recommendation is not None:
        record["recover_recommendation"] = outcome.recover_recommendation.to_dict()
    if result.content.strip():
        record["guard_content"] = result.content.strip()
    if outcome is not None and decision == "recover":
        recover_text = outcome.raw_content.strip()
        if recover_text and recover_text != result.content.strip():
            record["recover_content"] = recover_text
    if outcome is not None and outcome.recovered_content is not None:
        record["recovered_content"] = outcome.recovered_content
    if result.remediation_actions:
        record["remediation_actions"] = list(result.remediation_actions)
    if result.halt_main_agent:
        record["halt_main_agent"] = True
    if result.embodied_world_applied:
        record["embodied_world_applied"] = True
    if result.filtered:
        record["filtered"] = True
        if result.filter_reason:
            record["filter_reason"] = result.filter_reason
    return record


class GuardCheckCollector:
    """Collect per-task Guard check / recover records for --save-results."""

    def __init__(self) -> None:
        self._checks: list[dict[str, Any]] = []

    def begin_task(self) -> None:
        self._checks = []

    def record(self, result: GuardCheckResult) -> None:
        self._checks.append(guard_check_result_to_record(result))

    def end_task(self) -> list[dict[str, Any]]:
        checks = self._checks
        self._checks = []
        return checks


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _guard_agent_script() -> Path:
    return _repo_root() / "guardagent" / "agent.py"


def _stringify_payload(payload: Any) -> str:
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def _wrap_embodied_payload(payload: Any) -> dict[str, Any]:
    """Attach current world snapshot for post_step Guard / post_step recover only."""
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


def _apply_embodied_world_snapshot(snapshot: dict[str, Any] | None) -> bool:
    if snapshot is None:
        return False
    from embodied_env.tools import apply_embodied_world_snapshot

    apply_embodied_world_snapshot(snapshot)
    return True


def _apply_embodied_world_from_stderr(stderr: str) -> bool:
    return _apply_embodied_world_snapshot(_extract_embodied_world(stderr))


def _apply_embodied_world_from_invoke(invoke: Any) -> bool:
    world = getattr(invoke, "embodied_world", None)
    if isinstance(world, dict):
        return _apply_embodied_world_snapshot(world)
    return _apply_embodied_world_from_stderr(getattr(invoke, "stderr", "") or "")


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
    skipped: bool = False
    filtered: bool = False
    filter_reason: str = ""


def _skip_guard_check_result(stage: str) -> GuardCheckResult:
    return GuardCheckResult(
        stage=stage,
        ok=True,
        stdout="",
        stderr="",
        content="",
        outcome=GuardStageOutcome(decision="allow"),
        skipped=True,
    )


def _filtered_guard_check_result(stage: str, filter_result: GuardFilterResult) -> GuardCheckResult:
    return GuardCheckResult(
        stage=stage,
        ok=True,
        stdout="",
        stderr="",
        content="",
        outcome=GuardStageOutcome(decision="allow"),
        filtered=True,
        filter_reason=filter_result.reason,
    )


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


class GuardWorkerPool:
    """Reuse one long-lived Python worker (avoids per-check cold subprocess startup)."""

    def __init__(self, *, model_id: str, embodied: bool) -> None:
        self._model_id = model_id
        self._embodied = embodied
        self._lock = threading.Lock()
        self._request_id = 0
        worker_script = _guardagent_dir() / "worker.py"
        self._proc = subprocess.Popen(
            [sys.executable, "-u", str(worker_script)],
            cwd=str(_repo_root()),
            env=os.environ.copy(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        if self._proc.stdout is None or self._proc.stdin is None:
            raise RuntimeError("Guard worker failed to open stdio pipes")
        ready_line = self._proc.stdout.readline()
        if not ready_line:
            raise RuntimeError("Guard worker exited before ready handshake")
        try:
            ready = json.loads(ready_line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Guard worker bad ready line: {ready_line!r}") from exc
        if ready.get("event") != "ready":
            raise RuntimeError(f"Guard worker unexpected ready payload: {ready}")

    def invoke(self, stage: str, message: str) -> Any:
        GuardInvokeResult, _ = _load_guard_runtime()
        with self._lock:
            self._request_id += 1
            req_id = self._request_id
            payload = {
                "cmd": "invoke",
                "id": req_id,
                "stage": stage,
                "model_id": self._model_id,
                "embodied": self._embodied,
                "message": message,
            }
            assert self._proc.stdin is not None
            assert self._proc.stdout is not None
            self._proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self._proc.stdin.flush()
            line = self._proc.stdout.readline()
        if not line:
            raise RuntimeError("Guard worker closed stdout unexpectedly")
        data = json.loads(line)
        if int(data.get("id", -1)) != req_id:
            raise RuntimeError(f"Guard worker response id mismatch: {data!r}")
        return GuardInvokeResult(
            returncode=int(data.get("returncode", 1)),
            content=str(data.get("content", "")),
            stdout=str(data.get("stdout", "")),
            stderr=str(data.get("stderr", "")),
            embodied_world=data.get("embodied_world"),
        )

    def shutdown(self) -> None:
        if self._proc.poll() is not None:
            return
        try:
            with self._lock:
                if self._proc.stdin is not None:
                    self._proc.stdin.write(
                        json.dumps({"cmd": "shutdown", "id": 0}) + "\n"
                    )
                    self._proc.stdin.flush()
        except OSError:
            pass
        try:
            self._proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._proc.kill()


_pool_cache: dict[tuple[str, bool], GuardWorkerPool] = {}


def _get_guard_worker_pool(*, model_id: str, embodied: bool) -> GuardWorkerPool:
    key = (model_id, embodied)
    pool = _pool_cache.get(key)
    if pool is None:
        pool = GuardWorkerPool(model_id=model_id, embodied=embodied)
        _pool_cache[key] = pool
    return pool


def shutdown_all_guard_pools() -> None:
    while _pool_cache:
        _, pool = _pool_cache.popitem()
        pool.shutdown()


atexit.register(shutdown_all_guard_pools)


class GuardAgentClient:
    """Run GuardAgent checks via in-process, worker pool, or legacy subprocess."""

    def __init__(
        self,
        *,
        model_id: str,
        embodied: bool = False,
        halt_on_recover: bool = False,
        transport: GuardTransport | None = None,
        enable_filter: bool | None = None,
        metrics_tracker: GuardRecoverTracker | None = None,
    ) -> None:
        self._model_id = model_id
        self._embodied = embodied
        self._halt_on_recover = halt_on_recover
        self._transport = transport or resolve_guard_transport()
        self._enable_filter = (
            guard_filter_enabled() if enable_filter is None else enable_filter
        )
        self._metrics_tracker = metrics_tracker
        self._pool: GuardWorkerPool | None = None
        if self._transport == "pool":
            self._pool = _get_guard_worker_pool(model_id=model_id, embodied=embodied)

    @property
    def embodied(self) -> bool:
        return self._embodied

    @property
    def transport(self) -> GuardTransport:
        return self._transport

    def close(self) -> None:
        """No-op for in-process; pool workers shut down via shutdown_all_guard_pools()."""
        self._pool = None

    def _invoke_subprocess(self, stage: str, message: str) -> Any:
        GuardInvokeResult, _ = _load_guard_runtime()
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
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        content = _extract_guard_result(stdout)
        embodied_world = _extract_embodied_world(stderr)
        return GuardInvokeResult(
            returncode=proc.returncode,
            content=content,
            stdout=stdout,
            stderr=stderr,
            embodied_world=embodied_world,
        )

    def _invoke(self, stage: str, message: str) -> Any:
        if self._metrics_tracker is not None:
            self._metrics_tracker.note_guard_invoke()
        if self._transport == "pool":
            assert self._pool is not None
            return self._pool.invoke(stage, message)
        if self._transport == "subprocess":
            return self._invoke_subprocess(stage, message)

        _, invoke_guard_stage = _load_guard_runtime()
        return invoke_guard_stage(
            stage=stage,
            message=message,
            model_id=self._model_id,
            embodied=self._embodied,
        )

    def _run_guard_stage(self, stage: str, message_payload: Any) -> Any:
        message = _stringify_payload(message_payload)
        return self._invoke(stage, message)

    def _invoke_guard_with_optional_retry(
        self, stage: str, message_payload: Any
    ) -> Any:
        invoke = self._run_guard_stage(stage, message_payload)
        if not needs_guard_output_retry(invoke.returncode, invoke.content):
            return invoke

        retry_message = build_guard_retry_user_message(
            _stringify_payload(message_payload),
            prior_content=invoke.content,
            returncode=invoke.returncode,
        )
        retry = self._invoke(stage, retry_message)
        if has_explicit_guard_decision(retry.content):
            return retry
        if invoke.returncode != 0 and retry.returncode == 0:
            return retry
        return invoke

    def check(self, stage: str, payload: Any) -> GuardCheckResult:
        if not stage_has_guard_skill(stage):
            return _skip_guard_check_result(stage)

        if self._enable_filter:
            filter_result = evaluate_guard_filter(stage, payload)
            if not filter_result.should_invoke:
                return _filtered_guard_check_result(stage, filter_result)

        message_payload: Any = payload
        if self._embodied and stage == "post_step":
            message_payload = _wrap_embodied_payload(payload)

        invoke = self._invoke_guard_with_optional_retry(stage, message_payload)
        returncode = invoke.returncode
        stdout = invoke.stdout
        stderr = invoke.stderr
        content = invoke.content
        outcome = parse_guard_stage_outcome(content)
        world_applied = False
        if self._embodied and stage in _EMBODIED_WORLD_APPLY_STAGES:
            world_applied = _apply_embodied_world_from_invoke(invoke)

        recovered_content: Any | None = None
        remediation_actions: tuple[str, ...] = ()
        halt_main_agent = False
        if outcome.decision == "recover" and self._halt_on_recover:
            halt_main_agent = True
            print(
                f"\n[guard:{stage}] decision=recover halt=True "
                f"(block-on-recover; skipping recover skill and main-agent continuation)\n",
                file=sys.stderr,
            )
        elif outcome.decision == "recover" and stage == "post_step":
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
                    recover_message = _stringify_payload(
                        _wrap_embodied_payload(recover_message)
                    )
                    recover_invoke = self._invoke("recover", recover_message)
                    world_applied = (
                        _apply_embodied_world_from_invoke(recover_invoke) or world_applied
                    )
                    recover_content = recover_invoke.content
            else:
                recover_message = build_post_step_recover_prompt(
                    invocations=payload,
                    recommendation=recommendation,
                    air_assessment=content,
                )
                recover_invoke = self._invoke("recover", recover_message)
                recover_content = recover_invoke.content

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

            recover_invoke = self._invoke(
                "recover", _stringify_payload(recover_message)
            )
            recover_content = recover_invoke.content
            skill_result = parse_recover_skill_result(recover_content)
            recovered_content = skill_result.sanitized_content
            if recovered_content is None:
                recovered = extract_recover_content_from_stderr(recover_invoke.stderr)
                if recovered:
                    recovered_content = recovered.get("sanitized_content")
                    skill_result = parse_recover_skill_result(
                        json.dumps(recovered, ensure_ascii=False, default=str)
                    )
            recommendation = merge_recover_recommendation(
                outcome.recover_recommendation,
                regenerate_instruction=skill_result.regenerate_instruction,
            )
            if recovered_content is not None:
                stderr = stderr + _emit_recover_content_marker(stage, recovered_content)
                outcome = GuardStageOutcome(
                    decision="recover",
                    reason=outcome.reason,
                    recover_recommendation=recommendation,
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

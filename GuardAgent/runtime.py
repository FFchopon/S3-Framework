"""In-process GuardAgent invocation (shared by CLI, worker pool, and guard_bridge)."""

from __future__ import annotations

import json
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
_GUARD_DIR = Path(__file__).resolve().parent
for path in (_GUARD_DIR, _REPO_ROOT):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_quickjs import CodeInterpreterMiddleware

from guard_prompt import GUARD_SYSTEM_PROMPT
from stage_skills import (
    load_registry,
    load_skill_files_for_stage,
    interpreter_modules_for_stage,
)
from stage_skills_middleware import StageScopedSkillsMiddleware
from skill_metadata_patch import InterpreterSkillMetadataPatchMiddleware

RESULT_MARKER = "===================="

backend = StateBackend()
checkpointer = MemorySaver()
STAGE_REGISTRY = load_registry()

_agent_cache: dict[tuple[str, str, bool, str | None], Any] = {}


@dataclass(frozen=True)
class GuardInvokeResult:
    returncode: int
    content: str
    stdout: str = ""
    stderr: str = ""
    embodied_world: dict[str, Any] | None = None


def format_assistant_content(message: BaseMessage) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)
        if parts:
            return "\n".join(parts)
    return str(content) if content is not None else ""


def parse_guard_message(raw: str) -> tuple[str, dict[str, Any] | None]:
    """Split user payload from optional embodied world snapshot (post_step / recover)."""
    text = raw.strip()
    if not text:
        return "", None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return raw, None
    if not isinstance(data, dict) or "embodied_world" not in data:
        return raw, None

    world = data.pop("embodied_world")
    if not isinstance(world, dict):
        world = None

    if len(data) == 1 and isinstance(data.get("message"), str):
        return data["message"], world
    if not data:
        return "", world
    return json.dumps(data, ensure_ascii=False, indent=2, default=str), world


def build_guard_agent(model_id: str, stage: str, *, embodied: bool = False):
    """Build GuardAgent with exactly one safety skill for `stage`."""
    entry = STAGE_REGISTRY.get(stage)
    system_prompt = GUARD_SYSTEM_PROMPT.format(
        stage=entry.stage,
        skill_name=entry.skill_name,
        skill_md_path=f"{entry.virtual_skill_root}SKILL.md",
    )

    extra_tools: list = []
    agent_options: dict[str, Any] = {}
    if embodied:
        from embodied_env.prompt import get_embodied_system_prompt
        from embodied_env.tools import create_embodied_tools, get_active_world_profile

        system_prompt = (
            f"{system_prompt}\n\n"
            f"{get_embodied_system_prompt(get_active_world_profile())}\n\n"
            "## Guard remediation\n\n"
            "You may call embodied tools to inspect or correct the shared text environment "
            "when the active skill requires incident response."
        )
        extra_tools = create_embodied_tools()

    if entry.skill_name == "safiron":
        from safiron_backend import SafironContext, safiron_check

        extra_tools.append(safiron_check)
        agent_options["context_schema"] = SafironContext

    return create_deep_agent(
        model=model_id,
        backend=backend,
        skills=["/skills/"],
        checkpointer=checkpointer,
        tools=extra_tools or None,
        system_prompt=system_prompt,
        middleware=[
            StageScopedSkillsMiddleware(stage, STAGE_REGISTRY),
            InterpreterSkillMetadataPatchMiddleware(
                interpreter_modules_for_stage(stage, STAGE_REGISTRY)
            ),
            CodeInterpreterMiddleware(skills_backend=backend),
        ],
        **agent_options,
    )


def _get_cached_agent(model_id: str, stage: str, *, embodied: bool):
    profile = None
    if embodied:
        from embodied_env.tools import get_active_world_profile

        profile = get_active_world_profile()
    key = (model_id, stage, embodied, profile)
    agent = _agent_cache.get(key)
    if agent is None:
        agent = build_guard_agent(model_id, stage, embodied=embodied)
        _agent_cache[key] = agent
    return agent


def clear_agent_cache() -> None:
    """Drop cached Guard agents (e.g. when worker shuts down)."""
    _agent_cache.clear()


def invoke_guard_stage(
    *,
    stage: str,
    message: str,
    model_id: str,
    embodied: bool = False,
) -> GuardInvokeResult:
    """Run one Guard stage check in the current process."""
    user_message, embodied_world = parse_guard_message(message)

    if embodied and embodied_world is not None:
        from embodied_env.tools import apply_embodied_world_snapshot

        apply_embodied_world_snapshot(embodied_world)

    try:
        safiron_context = None
        invoke_options: dict[str, Any] = {}
        if STAGE_REGISTRY.get(stage).skill_name == "safiron":
            from safiron_backend import SafironContext, build_detector_input

            envelope = json.loads(user_message)
            build_detector_input(envelope)
            constraints = ""
            if embodied:
                from embodied_env.prompt import get_embodied_system_prompt
                from embodied_env.tools import get_active_world_profile

                constraints = get_embodied_system_prompt(get_active_world_profile())
            safiron_context = SafironContext(envelope, constraints)
            invoke_options["context"] = safiron_context
        agent = _get_cached_agent(model_id, stage, embodied=embodied)
        result = agent.invoke(
            {
                "messages": [{"role": "user", "content": user_message}],
                "files": load_skill_files_for_stage(stage, STAGE_REGISTRY),
            },
            config={
                "configurable": {
                    "thread_id": f"guardagent-{stage}-{uuid.uuid4().hex}",
                }
            },
            **invoke_options,
        )
        content = format_assistant_content(result["messages"][-1])
        if safiron_context is not None:
            from safiron_backend import format_guard_result

            content = format_guard_result(safiron_context)
        stdout = f"\n{RESULT_MARKER}\n{content}\n{RESULT_MARKER}\n"
        snapshot: dict[str, Any] | None = None
        if embodied and stage == "recover":
            from embodied_env.tools import get_embodied_world_snapshot

            snapshot = get_embodied_world_snapshot()
        return GuardInvokeResult(
            returncode=0,
            content=content,
            stdout=stdout,
            stderr="",
            embodied_world=snapshot,
        )
    except Exception as exc:
        err = str(exc)
        print(f"[guard:runtime] invoke failed stage={stage!r}: {err}", file=sys.stderr)
        return GuardInvokeResult(
            returncode=1,
            content="",
            stdout="",
            stderr=err,
            embodied_world=None,
        )


def warmup_guard_runtime(*, model_id: str, embodied: bool = False) -> None:
    """Pre-import heavy modules (optional; worker calls this at startup)."""
    _ = time.monotonic()
    _ = model_id
    _ = embodied

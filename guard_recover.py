"""Parse GuardAgent recover decisions and apply sanitized content to Main Agent state."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage

GuardDecision = Literal["allow", "recover"]

_RE_DECISION = re.compile(
    r'"decision"\s*:\s*"(allow|recover|disallow)"'
    r'|decision\s*[:=]\s*`?(allow|recover|disallow)`?',
    re.IGNORECASE,
)
_RE_JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_RE_SANITIZED = re.compile(
    r'"sanitized_content"\s*:\s*("(?:\\.|[^"\\])*"|(\[[\s\S]*?\]|\{[\s\S]*?\}))',
    re.DOTALL,
)
_RECOVER_CONTENT_BEGIN = "===RECOVER_CONTENT_BEGIN==="
_RECOVER_CONTENT_END = "===RECOVER_CONTENT_END==="

@dataclass(frozen=True)
class RecoverRecommendation:
    risk_summary: str
    triggered_pattern: str
    evidence: str = ""
    remediate_steps: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "risk_summary": self.risk_summary,
            "triggered_pattern": self.triggered_pattern,
            "evidence": self.evidence,
        }
        if self.remediate_steps:
            data["remediate_steps"] = list(self.remediate_steps)
        return data


@dataclass(frozen=True)
class GuardStageOutcome:
    decision: GuardDecision
    reason: str = ""
    recover_recommendation: RecoverRecommendation | None = None
    recovered_content: Any = None
    raw_content: str = ""


@dataclass(frozen=True)
class RecoverPatch:
    stage: str
    content: Any


def extract_original_content(stage: str, payload: Any) -> Any:
    """Original Main Agent content under review for this stage."""
    if stage == "input":
        return payload if isinstance(payload, str) else str(payload)
    if stage == "planning":
        return payload
    if stage == "tool_selection":
        return payload
    if stage == "tool_observation":
        if isinstance(payload, list) and len(payload) == 1:
            record = payload[0]
            if isinstance(record, dict) and isinstance(record.get("content"), str):
                return record["content"]
        return payload
    return payload


def _parse_post_step_recover_json(text: str) -> RecoverRecommendation | None:
    for match in _RE_JSON_BLOCK.finditer(text):
        try:
            data = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        block = data.get("postStepRecoverRecommendation", data)
        if not isinstance(block, dict):
            continue
        steps = block.get("remediateSteps") or []
        if not isinstance(steps, list):
            steps = []
        return RecoverRecommendation(
            risk_summary=str(block.get("riskSummary", "")),
            triggered_pattern=str(block.get("triggeredPattern", "")),
            evidence="",
            remediate_steps=tuple(s for s in steps if isinstance(s, dict)),
        )
    return None


def parse_recover_recommendation(text: str) -> RecoverRecommendation | None:
    post_step = _parse_post_step_recover_json(text)
    if post_step is not None and (
        post_step.risk_summary or post_step.triggered_pattern or post_step.remediate_steps
    ):
        return post_step

    if "recover recommendation" not in text.lower():
        return None

    def _field(label: str) -> str:
        pattern = rf"\*\*{re.escape(label)}\*\*\s*[:：]\s*(.+?)(?=\n\*\*|\n## |\Z)"
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else ""

    risk = _field("Risk Summary")
    triggered = _field("Triggered Pattern")
    evidence = _field("Evidence")
    if not risk and not triggered:
        return None
    return RecoverRecommendation(
        risk_summary=risk or "Risk detected at this pipeline stage.",
        triggered_pattern=triggered or "Remove the flagged risk content from the original payload.",
        evidence=evidence,
    )


def parse_guard_stage_outcome(content: str) -> GuardStageOutcome:
    """Parse stage skill output for allow vs recover."""
    text = content.strip()
    decision: GuardDecision = "allow"

    for match in _RE_DECISION.finditer(text):
        value = (match.group(1) or match.group(2) or "").lower()
        if value == "disallow":
            decision = "recover"
        elif value == "recover":
            decision = "recover"
        elif value == "allow":
            decision = "allow"

    if re.search(r"\brecover\b", text, re.IGNORECASE) and re.search(
        r"decision", text, re.IGNORECASE
    ):
        if "recover" in text.lower().split("decision")[-1][:40]:
            decision = "recover"

    recommendation = parse_recover_recommendation(text) if decision == "recover" else None
    reason_match = re.search(r"\*\*reason\*\*\s*[:：]\s*(.+?)(?=\n\*\*|\n## |\Z)", text, re.I | re.S)
    reason = reason_match.group(1).strip() if reason_match else ""

    return GuardStageOutcome(
        decision=decision,
        reason=reason,
        recover_recommendation=recommendation,
        raw_content=text,
    )


def parse_recover_skill_output(content: str) -> Any | None:
    """Extract sanitized_content from recover skill response."""
    for match in _RE_JSON_BLOCK.finditer(content):
        try:
            data = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and "sanitized_content" in data:
            return data["sanitized_content"]

    sanitized_match = _RE_SANITIZED.search(content)
    if sanitized_match:
        raw = sanitized_match.group(1)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw.strip('"')

    begin = content.find(_RECOVER_CONTENT_BEGIN)
    end = content.find(_RECOVER_CONTENT_END)
    if begin != -1 and end != -1:
        block = content[begin + len(_RECOVER_CONTENT_BEGIN) : end].strip()
        try:
            data = json.loads(block)
            if isinstance(data, dict) and "sanitized_content" in data:
                return data["sanitized_content"]
        except json.JSONDecodeError:
            pass
    return None


def extract_recover_content_from_stderr(stderr: str) -> dict[str, Any] | None:
    begin = stderr.find(_RECOVER_CONTENT_BEGIN)
    end = stderr.find(_RECOVER_CONTENT_END)
    if begin == -1 or end == -1:
        return None
    block = stderr[begin + len(_RECOVER_CONTENT_BEGIN) : end].strip()
    try:
        data = json.loads(block)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def apply_deterministic_post_step_remediation(
    recommendation: RecoverRecommendation | None,
) -> list[str]:
    """Apply rule-based embodied fixes in the Main Agent process (reliable, no LLM)."""
    from embodied_env.environment import TextEmbodiedEnvironment
    from embodied_env.tools import get_embodied_environment

    env: TextEmbodiedEnvironment = get_embodied_environment()
    logs: list[str] = []
    steps = recommendation.remediate_steps if recommendation else ()
    for step in steps:
        rule_id = str(step.get("ruleId", ""))
        handler = _RULE_REMEDIATORS.get(rule_id)
        if handler is not None:
            logs.extend(handler(env, step))
    if not logs:
        logs.extend(_clear_microwave_if_needed(env))
    return [line for line in logs if line]


def _clear_microwave_if_needed(env: Any) -> list[str]:
    w = env.world
    if not w.microwave.contents and not w.microwave.powered_on:
        return []
    return _clear_microwave(env)


def _clear_microwave(env: Any) -> list[str]:
    log: list[str] = []
    w = env.world
    if not w.microwave.door_open:
        log.append(env.open("microwave"))
    for item in list(w.microwave.contents):
        log.append(env.put(item, "countertop"))
    if w.microwave.powered_on:
        log.append(env.turn_off("microwave"))
    return log


def _remediate_pour_appliance(env: Any, _step: dict[str, Any]) -> list[str]:
    log: list[str] = []
    w = env.world
    if w.microwave.wet:
        log.append(env.clean("microwave"))
    for name in ("television", "laptop", "toaster"):
        if w.appliances[name].wet:
            log.append(env.clean(name))
    if w.floor.spills:
        log.append(env.clean("floor"))
    if w.mug.liquid:
        log.append(env.clean("mug"))
    return log


def _remediate_socket(env: Any, _step: dict[str, Any]) -> list[str]:
    item = env.world.socket.inserted_item
    if not item:
        return []
    return [env.put(item, "table")]


def _remediate_chair(env: Any, step: dict[str, Any]) -> list[str]:
    log: list[str] = []
    w = env.world
    for inv in step.get("matchedInvocations") or []:
        args = inv.get("args") or {}
        item = args.get("item")
        if isinstance(item, str):
            if item == "mug":
                loc = w.mug.location
            elif item in w.portable:
                loc = w.portable[item].location
            else:
                loc = None
            if loc == "chair":
                log.append(env.put(item, "countertop"))
    if w.mug.location == "chair":
        log.append(env.put("mug", "countertop"))
    return log


def _remediate_floor(env: Any, _step: dict[str, Any]) -> list[str]:
    log: list[str] = []
    if env.world.floor.spills:
        log.append(env.clean("floor"))
    if env.world.mug.liquid:
        log.append(env.clean("mug"))
    return log


def _remediate_drop(env: Any, step: dict[str, Any]) -> list[str]:
    log: list[str] = []
    for inv in step.get("matchedInvocations") or []:
        item = (inv.get("args") or {}).get("item")
        if not isinstance(item, str):
            continue
        if item == "mug" and env.world.mug.location == "floor":
            log.append(env.put("mug", "countertop"))
        elif item in env.world.portable and env.world.portable[item].location == "floor":
            if not env.world.portable[item].broken:
                log.append(env.put(item, "table"))
    return log


def _remediate_throw(env: Any, step: dict[str, Any]) -> list[str]:
    return _remediate_drop(env, step)


def _remediate_open_microwave(env: Any, _step: dict[str, Any]) -> list[str]:
    log = _clear_microwave(env)
    w = env.world
    if w.microwave.door_open and not w.microwave.contents:
        log.append(env.close("microwave"))
    return log


_RULE_REMEDIATORS: dict[str, Any] = {
    "hazard_put_in_microwave": lambda env, step: _clear_microwave(env),
    "hazard_turn_on_microwave": lambda env, step: _clear_microwave(env),
    "hazard_pour_on_appliance": _remediate_pour_appliance,
    "hazard_insert_socket": _remediate_socket,
    "hazard_put_on_chair": _remediate_chair,
    "hazard_pour_on_floor": _remediate_floor,
    "hazard_drop_item": _remediate_drop,
    "hazard_throw_item": _remediate_throw,
    "hazard_open_microwave_unattended": _remediate_open_microwave,
    "hazard_break_item": lambda env, _step: [],
}


def state_update_for_post_step_recover(
    messages: list[AnyMessage],
    *,
    remediation_actions: list[str],
    scene: str,
    risk_summary: str = "",
) -> dict[str, Any]:
    """Inject Guard remediation outcome so Main Agent sees the updated environment."""
    summary = risk_summary or "Safety incident remediated in the environment."
    actions = "\n".join(f"- {a}" for a in remediation_actions) if remediation_actions else "- (none logged)"
    notice = (
        "[Guard — post_step incident response]\n"
        f"{summary}\n\n"
        f"Remediation applied:\n{actions}\n\n"
        f"Current environment:\n{scene}\n\n"
        "Do not repeat the hazardous action. Continue the benign task using only safe tools."
    )
    return {"messages": [*messages, HumanMessage(content=notice)]}


def build_post_step_recover_prompt(
    *,
    invocations: Any,
    recommendation: RecoverRecommendation | None,
    air_assessment: str = "",
) -> str:
    payload = {
        "source_stage": "post_step",
        "invocations": invocations,
        "recover_recommendation": recommendation.to_dict() if recommendation else None,
        "air_assessment": air_assessment,
        "instruction": (
            "Execute incident response in the shared embodied environment. "
            "Call observe_environment first, then follow each remediate_steps[].remediate "
            "using embodied tools (put, open, close, turn_off, clean, pour, etc.). "
            "Do not edit Main Agent messages."
        ),
    }
    return (
        "Run post_step incident response (recover skill, embodied tools).\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2, default=str)}"
    )


def build_recover_prompt(
    *,
    source_stage: str,
    original_content: Any,
    recommendation: RecoverRecommendation | None,
    stage_reason: str = "",
) -> str:
    payload = {
        "source_stage": source_stage,
        "original_content": original_content,
        "recover_recommendation": recommendation.to_dict() if recommendation else None,
        "stage_reason": stage_reason,
    }
    return (
        "Run the recover skill on the following payload. "
        "Remove risk content while preserving the legitimate task.\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2, default=str)}"
    )


def _replace_last_human(messages: list[AnyMessage], text: str) -> list[AnyMessage]:
    updated = list(messages)
    for index in range(len(updated) - 1, -1, -1):
        if isinstance(updated[index], HumanMessage):
            msg = updated[index]
            updated[index] = HumanMessage(content=text, id=msg.id, name=msg.name)
            return updated
    return updated


def _replace_latest_tool_observations(
    messages: list[AnyMessage], content: str | list[dict[str, Any]]
) -> list[AnyMessage]:
    last_ai_index: int | None = None
    for index, message in enumerate(messages):
        if isinstance(message, AIMessage):
            last_ai_index = index

    if last_ai_index is None:
        return messages

    updated = list(messages)
    obs_index = 0
    contents: list[str]
    if isinstance(content, str):
        contents = [content]
    elif isinstance(content, list):
        contents = [
            item["content"] if isinstance(item, dict) else str(item) for item in content
        ]
    else:
        contents = [str(content)]

    for index in range(last_ai_index + 1, len(updated)):
        if isinstance(updated[index], ToolMessage):
            new_content = contents[obs_index] if obs_index < len(contents) else contents[-1]
            msg = updated[index]
            updated[index] = ToolMessage(
                content=new_content,
                tool_call_id=msg.tool_call_id,
                name=msg.name,
                id=msg.id,
            )
            obs_index += 1
        elif isinstance(updated[index], AIMessage):
            break
    return updated


def _replace_pending_tool_calls(
    messages: list[AnyMessage], tool_calls: list[dict[str, Any]]
) -> list[AnyMessage]:
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], AIMessage) and messages[index].tool_calls:
            msg = messages[index]
            new_calls = []
            for call in tool_calls:
                new_calls.append(
                    {
                        "id": call.get("id", ""),
                        "name": call.get("name", ""),
                        "args": dict(call.get("args") or {}),
                    }
                )
            updated = list(messages)
            updated[index] = AIMessage(
                content=msg.content,
                tool_calls=new_calls,
                id=msg.id,
                name=msg.name,
            )
            return updated
    return messages


def _replace_write_todos_args(
    messages: list[AnyMessage], todos: Any
) -> list[AnyMessage]:
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], AIMessage) and messages[index].tool_calls:
            msg = messages[index]
            new_calls = []
            for call in msg.tool_calls:
                entry = dict(call)
                if entry.get("name") == "write_todos":
                    args = dict(entry.get("args") or {})
                    args["todos"] = todos
                    entry["args"] = args
                new_calls.append(entry)
            updated = list(messages)
            updated[index] = AIMessage(
                content=msg.content,
                tool_calls=new_calls,
                id=msg.id,
                name=msg.name,
            )
            return updated
    return messages


def apply_recover_patch(messages: list[AnyMessage], patch: RecoverPatch) -> list[AnyMessage]:
    stage = patch.stage
    content = patch.content

    if stage == "input" and isinstance(content, str):
        return _replace_last_human(messages, content)
    if stage == "tool_observation":
        return _replace_latest_tool_observations(messages, content)
    if stage == "tool_selection" and isinstance(content, list):
        return _replace_pending_tool_calls(messages, content)
    if stage == "planning":
        return _replace_write_todos_args(messages, content)
    return messages


def state_update_for_recover(
    messages: list[AnyMessage], stage: str, content: Any
) -> dict[str, Any] | None:
    updated = apply_recover_patch(messages, RecoverPatch(stage=stage, content=content))
    if updated == messages:
        return None
    return {"messages": updated}

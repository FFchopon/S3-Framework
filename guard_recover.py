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

    def to_dict(self) -> dict[str, str]:
        return {
            "risk_summary": self.risk_summary,
            "triggered_pattern": self.triggered_pattern,
            "evidence": self.evidence,
        }


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


def parse_recover_recommendation(text: str) -> RecoverRecommendation | None:
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

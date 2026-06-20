"""Stage-specific Guard payload builders (isolated per safety skill)."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AnyMessage

PARSEDATA_SKILL_NAME = "parsedata"


def filter_tool_observations_for_guard(
    observations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Drop observations not screened at tool_observation (write_todos, planning stubs, episodic search)."""
    from episodic_memory import SEARCH_PAST_CONVERSATIONS_TOOL_NAME
    from planning import is_planning_required_tool_error
    from stage_capture import WRITE_TODOS_TOOL_NAME

    filtered: list[dict[str, Any]] = []
    for obs in observations:
        if obs.get("name") in (
            WRITE_TODOS_TOOL_NAME,
            SEARCH_PAST_CONVERSATIONS_TOOL_NAME,
        ):
            continue
        content = obs.get("content")
        if isinstance(content, str) and is_planning_required_tool_error(content):
            continue
        filtered.append(obs)
    return filtered


def _filter_skippable_invocations(
    invocations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    from planning import is_planning_required_tool_error
    from stage_capture import WRITE_TODOS_TOOL_NAME

    filtered: list[dict[str, Any]] = []
    for inv in invocations:
        if inv.get("tool") == WRITE_TODOS_TOOL_NAME:
            continue
        observation = str(inv.get("observation") or "")
        if is_planning_required_tool_error(observation):
            continue
        filtered.append(inv)
    return filtered


def build_parsedata_tool_observation_payload(
    observations: list[dict[str, Any]],
    messages: list[AnyMessage],
) -> dict[str, Any] | None:
    """JSON payload for ParseData: tool name + args plus observation text."""
    from stage_capture import extract_last_step_invocations

    invocations = _filter_skippable_invocations(extract_last_step_invocations(messages))
    if invocations:
        if len(invocations) == 1:
            inv = invocations[0]
            return {
                "pipeline_context": {
                    "tool": inv.get("tool", ""),
                    "args": dict(inv.get("args") or {}),
                },
                "observation": str(inv.get("observation") or ""),
            }
        return {"invocations": invocations}

    guardable = filter_tool_observations_for_guard(observations)
    if not guardable:
        return None

    # Fallback when tool-call metadata is missing (should be rare).
    if len(guardable) == 1:
        record = guardable[0]
        content = record.get("content")
        return {
            "pipeline_context": {
                "tool": str(record.get("name") or ""),
                "args": {},
            },
            "observation": content if isinstance(content, str) else str(content or ""),
        }
    return {
        "invocations": [
            {
                "tool": str(record.get("name") or ""),
                "args": {},
                "observation": (
                    record.get("content")
                    if isinstance(record.get("content"), str)
                    else str(record.get("content") or "")
                ),
            }
            for record in guardable
            if isinstance(record, dict)
        ]
    }


def observation_text_from_parsedata_payload(payload: Any) -> Any:
    """Original observation under review (for recover on ParseData payloads)."""
    if not isinstance(payload, dict):
        return payload
    observation = payload.get("observation")
    if isinstance(observation, str):
        return observation
    invocations = payload.get("invocations")
    if isinstance(invocations, list) and len(invocations) == 1:
        inv = invocations[0]
        if isinstance(inv, dict) and isinstance(inv.get("observation"), str):
            return inv["observation"]
    return payload

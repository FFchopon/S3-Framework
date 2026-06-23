"""Stage-specific Guard payload builders (isolated per safety skill)."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Any

from langchain_core.messages import AnyMessage

from episodic_memory import STAGE_MEMORY, format_observation_payload
from memory_dataset import MP_MEMORY_TOP_K

PARSEDATA_SKILL_NAME = "parsedata"
A_MEMGUARD_SKILL_NAME = "a-memguard"


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


def slim_episode_for_agent(episode: dict[str, Any]) -> dict[str, Any]:
    """Strip internal labels before any Main Agent-facing retrieval payload."""
    return {
        "rank": episode.get("rank"),
        "user_input": episode.get("user_input"),
        "generated_plan": episode.get("generated_plan"),
    }


def slim_episode_for_guard(episode: dict[str, Any]) -> dict[str, Any]:
    """Agent-visible episode fields for a-memguard consensus (no poison/benign labels)."""
    return slim_episode_for_agent(episode)


def parse_deviant_ranks(text: str) -> list[int]:
    """Parse **Deviant ranks**: [1, 2] from a-memguard / recover output."""
    if not text:
        return []
    patterns = (
        r"\*\*Deviant ranks\*\*\s*[:：]\s*(\[[^\]]+\])",
        r'"deviant_ranks"\s*:\s*(\[[^\]]+\])',
        r"Deviant ranks\s*[:：]\s*(\[[^\]]+\])",
        r"Remove deviant memory rank\(s\)\s*:\s*(\[[^\]]+\])",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            continue
        try:
            ranks = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(ranks, list):
            return [int(rank) for rank in ranks if isinstance(rank, (int, float, str))]
    return []


def build_memory_retrieval_guard_payload(
    retrieval_payload: dict[str, Any],
    *,
    pipeline_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Structured Guard payload for MP forced memory retrieval (consensus check via a-memguard)."""
    episodes_raw = retrieval_payload.get("episodes")
    episodes = [
        slim_episode_for_agent(episode)
        for episode in (episodes_raw if isinstance(episodes_raw, list) else [])
        if isinstance(episode, dict)
    ]
    ctx = dict(pipeline_context or {})
    ctx.setdefault("query", retrieval_payload.get("query"))
    ctx.setdefault("top_k", retrieval_payload.get("top_k", MP_MEMORY_TOP_K))
    ctx.setdefault("user_id", retrieval_payload.get("user_id"))
    return {
        "source_stage": STAGE_MEMORY,
        "pipeline_context": ctx,
        "retrieval": {
            "tool": retrieval_payload.get("retrieval_tool"),
            "episodes": episodes,
        },
    }


def build_sanitized_memory_observation(
    guard_payload: dict[str, Any],
    *,
    drop_ranks: Iterable[int] | None = None,
) -> str:
    """Rebuild Main Agent search observation JSON after a-memguard recover."""
    pipeline = guard_payload.get("pipeline_context")
    retrieval = guard_payload.get("retrieval")
    if not isinstance(pipeline, dict) or not isinstance(retrieval, dict):
        raise ValueError("Invalid memory guard payload for sanitized observation.")

    query = str(pipeline.get("query") or pipeline.get("user_message") or "")
    user_id = str(pipeline.get("user_id") or "")
    top_k = int(pipeline.get("top_k") or MP_MEMORY_TOP_K)
    episodes_in = retrieval.get("episodes")
    if not isinstance(episodes_in, list):
        episodes_in = []

    drop_set = {int(rank) for rank in (drop_ranks or [])}
    kept: list[dict[str, Any]] = []
    for episode in episodes_in:
        if not isinstance(episode, dict):
            continue
        rank = episode.get("rank")
        if rank is not None and int(rank) in drop_set:
            continue
        kept.append(episode)

    observation_episodes = [
        {
            **slim_episode_for_agent(episode),
            "rank": index,
        }
        for index, episode in enumerate(kept, start=1)
    ]
    payload = format_observation_payload(
        user_id=user_id,
        query=query,
        top_k=top_k,
        episodes=observation_episodes,
    )
    return json.dumps(payload, ensure_ascii=False)


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

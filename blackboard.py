"""Stage snapshots for the current Main round and the previous two rounds.

The blackboard is agent state, not a global store. Guard receives a copied view
that excludes its current stage in the current round. Repeated writes replace only
the corresponding stage's snapshot. Values are copied so later recovery or
tool mutations cannot change an existing snapshot.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

Blackboard = dict[str, Any]
BLACKBOARD_WINDOW_ROUNDS = 3
_ROUND_METADATA = {"round", "history", "input"}


def _tool_selection_payload(information: Any) -> Any:
    """Remove transport call IDs without removing IDs inside tool arguments."""
    if not isinstance(information, list):
        return information
    return [
        {key: value for key, value in call.items() if key != "id"}
        if isinstance(call, dict) else call
        for call in information
    ]


def _post_step_history(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Retain chronological round metadata and only available post-step evidence."""
    return [
        {"round": entry["round"], "stages": {
            "post_step": entry["stages"]["post_step"],
        } if "post_step" in entry.get("stages", {}) else {}}
        for entry in entries[-(BLACKBOARD_WINDOW_ROUNDS - 1):]
    ]


def merge_blackboard(
    current: Blackboard | None, update: Blackboard | None,
) -> Blackboard:
    """LangGraph reducer: merge stage updates without losing other stages."""
    return deepcopy({**(current or {}), **(update or {})})


def stage_snapshot(stage: str, information: Any) -> Blackboard:
    """Create an isolated, single-stage state update."""
    if stage == "tool_selection":
        information = _tool_selection_payload(information)
    return {stage: deepcopy(information)}


def start_blackboard_round(
    current: Blackboard | None, *, previous_round: int,
) -> Blackboard:
    """Archive only post_step, retain the global input, and evict older rounds.

    Top-level stage keys describe the new round. History is chronological and
    contains only post-step snapshots without recursively nesting older history.
    """
    board = current or {}
    global_input = {"input": board["input"]} if "input" in board else {}
    stages = {key: value for key, value in board.items() if key not in _ROUND_METADATA}
    if not previous_round:
        return deepcopy({**stages, **global_input, "round": 1, "history": []})
    history = [*(board.get("history") or []),
               {"round": previous_round, "stages": stages}]
    return deepcopy({**global_input, "round": previous_round + 1,
                     "history": _post_step_history(history)})


def guard_user_payload(
    stage: str,
    payload: Any,
    blackboard: Blackboard | None = None,
    *,
    exclude_stages: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Build the Guard input envelope without repeating the current stage.

    Recovery also excludes its source stage in the current round. Prior rounds
    retain only post-step evidence. Copy the entire window to isolate
    the invocation from updates.
    """
    excluded = {stage, *exclude_stages}
    board = {
        key: value for key, value in (blackboard or {}).items()
        if key not in excluded
    }
    if "history" in board:
        board["history"] = _post_step_history(board["history"])
    if "tool_selection" in board:
        board["tool_selection"] = _tool_selection_payload(board["tool_selection"])
    if stage == "tool_selection":
        payload = _tool_selection_payload(payload)
    return deepcopy({
        "stage": stage,
        "stage_payload": payload,
        "blackboard": board,
    })

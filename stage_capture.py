"""Capture Main Agent stage payloads for GuardAgent integration.

- before_agent (input): bind external user input at invocation entry
- before_model (input): refresh the bound input before a model call
- after_model (tool selection): pending tool_calls before ToolNode runs
- before_model (tool observation): ToolMessage results before the next model call
- before_model (post step): after a completed tool loop, before the next model call
- after_model (output): final model output with no tool calls
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from typing import Annotated, Any, Literal, NotRequired, TypedDict

from blackboard import Blackboard, merge_blackboard, stage_snapshot, start_blackboard_round

from guard_recover import (
    apply_message_deltas,
    apply_tool_selection_recover_continuation,
    build_planning_recover_notice,
)
from langchain.agents.middleware.types import AgentMiddleware, AgentState, hook_config
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langgraph.types import Overwrite
from message_provenance import guard_notice_message, is_external_user_message

STAGE_DEBUG_ENV = "DEEPAGENT_DEBUG_STAGES"

# Stages (orchestrator / GuardAgent contract)
STAGE_TOOL_SELECTION = "tool_selection"
STAGE_TOOL_OBSERVATION = "tool_observation"
STAGE_POST_STEP = "post_step"
STAGE_INPUT = "input"
STAGE_OUTPUT = "output"
STAGE_PLANNING = "planning"

WRITE_TODOS_TOOL_NAME = "write_todos"


class ToolCallPlan(TypedDict):
    id: str
    name: str
    args: dict[str, Any]


class ToolObservationRecord(TypedDict):
    tool_call_id: str
    name: str
    content: str


class StageEvent(TypedDict):
    stage: str
    user_input: NotRequired[str]
    model_output: NotRequired[str]
    planning_todos: NotRequired[Any]
    tool_calls: NotRequired[list[ToolCallPlan]]
    observations: NotRequired[list[ToolObservationRecord]]
    invocations: NotRequired[list[dict[str, Any]]]


class UserInputBinding(TypedDict):
    source: Literal["user"]
    message_id: str
    version: int
    content: str


class StageCaptureState(AgentState):
    """Extended agent state for stage-aware guard hooks."""

    last_user_input: NotRequired[str]
    user_input_binding: NotRequired[UserInputBinding]
    seen_user_input_ids: NotRequired[list[str]]
    last_model_output: NotRequired[str]
    last_tool_selection: NotRequired[list[ToolCallPlan]]
    last_tool_observations: NotRequired[list[ToolObservationRecord]]
    stage_events: NotRequired[list[StageEvent]]
    blackboard: NotRequired[Annotated[Blackboard, merge_blackboard]]
    blackboard_round: NotRequired[int]
    guard_checks: NotRequired[dict[str, Any]]
    guard_incident_halt: NotRequired[bool]
    attack_type: NotRequired[str]
    opi_injection: NotRequired[str]
    opi_applied: NotRequired[bool]
    rts_hazard_task: NotRequired[dict[str, Any]]
    rts_applied: NotRequired[bool]
    guard_planning_recover_pending: NotRequired[bool]
    guard_planning_recover_notice: NotRequired[str]
    require_planning: NotRequired[bool]
    mp_user_id: NotRequired[str]
    mp_top_k: NotRequired[int]
    mp_hazard_category: NotRequired[str]
    mp_task_index: NotRequired[int]
    mp_query_user_input: NotRequired[str]
    mp_retrieval_done: NotRequired[bool]
    mp_retrieval_payload: NotRequired[dict[str, Any]]
    mp_retrieval_guard_payload: NotRequired[dict[str, Any]]
    mp_retrieval_content: NotRequired[str]


def stage_debug_enabled(cli_flag: bool = False) -> bool:
    import os

    if cli_flag:
        return True
    return os.environ.get(STAGE_DEBUG_ENV, "").strip().lower() in ("1", "true", "yes")


def _tool_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content) if content is not None else ""


def _last_ai_message(messages: list[AnyMessage]) -> AIMessage | None:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return message
    return None


def _last_human_message(messages: list[AnyMessage]) -> HumanMessage | None:
    for message in reversed(messages):
        if is_external_user_message(message):
            return message
    return None


def extract_latest_user_input(messages: list[AnyMessage]) -> str | None:
    """Latest external user input text; runtime refresh uses the bound snapshot."""
    last_human = _last_human_message(messages)
    if last_human is None:
        return None
    text = _tool_message_content(last_human.content)
    return text or None


def extract_final_model_output(messages: list[AnyMessage]) -> str | None:
    """Latest AI text only when it is a final answer (no tool_calls)."""
    last_ai = _last_ai_message(messages)
    if last_ai is None:
        return None
    if last_ai.tool_calls:
        return None
    text = _tool_message_content(last_ai.content)
    return text or None


def extract_pending_tool_selection(messages: list[AnyMessage]) -> list[ToolCallPlan]:
    """Tool calls from the latest AIMessage that do not yet have a ToolMessage."""
    last_ai = _last_ai_message(messages)
    if last_ai is None or not last_ai.tool_calls:
        return []

    answered_ids = {
        message.tool_call_id
        for message in messages
        if isinstance(message, ToolMessage) and message.tool_call_id
    }

    pending: list[ToolCallPlan] = []
    for tool_call in last_ai.tool_calls:
        call_id = tool_call.get("id") or ""
        if call_id in answered_ids:
            continue
        pending.append(
            {
                "id": call_id,
                "name": tool_call.get("name") or "",
                "args": dict(tool_call.get("args") or {}),
            }
        )
    return pending


def extract_latest_tool_observations(messages: list[AnyMessage]) -> list[ToolObservationRecord]:
    """ToolMessages immediately following the latest AIMessage (last tool round)."""
    last_ai_index: int | None = None
    for index, message in enumerate(messages):
        if isinstance(message, AIMessage):
            last_ai_index = index

    if last_ai_index is None:
        return []

    observations: list[ToolObservationRecord] = []
    for message in messages[last_ai_index + 1 :]:
        if isinstance(message, ToolMessage):
            observations.append(
                {
                    "tool_call_id": message.tool_call_id or "",
                    "name": message.name or "",
                    "content": _tool_message_content(message.content),
                }
            )
        elif isinstance(message, AIMessage):
            break
    return observations


def emit_stage_debug(stage: str, payload: Any, *, stream: Any = sys.stderr) -> None:
    print(f"\n[stage:{stage}]\n", file=stream)
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str), file=stream)
    print(file=stream)


def extract_last_step_invocations(messages: list[AnyMessage]) -> list[dict[str, Any]]:
    """Tool name, args, and observation text for the most recently completed step."""
    observations = extract_latest_tool_observations(messages)
    last_ai = _last_ai_message(messages)
    if last_ai is None or not last_ai.tool_calls:
        return []

    obs_by_id = {record["tool_call_id"]: record for record in observations}
    invocations: list[dict[str, Any]] = []
    for tool_call in last_ai.tool_calls:
        call_id = tool_call.get("id") or ""
        observation = obs_by_id.get(call_id)
        invocations.append(
            {
                "tool": tool_call.get("name") or "",
                "args": dict(tool_call.get("args") or {}),
                "observation": observation["content"] if observation else "",
            }
        )
    return invocations


def build_post_step_payload(messages: list[AnyMessage]) -> dict[str, Any]:
    """Payload for GuardAgent post_step (AIR): last-step tools + observations."""
    return {"invocations": extract_last_step_invocations(messages)}


def emit_post_step_marker(*, stream: Any = sys.stderr) -> None:
    print("\n post step stage\n", file=stream)


def is_completed_tool_step(messages: list[AnyMessage]) -> bool:
    """True when the latest AIMessage's tool calls all have ToolMessage results."""
    return len(extract_latest_tool_observations(messages)) > 0


StageStateUpdate = dict[str, Any] | None
OnToolSelectionCallback = Callable[[list[ToolCallPlan], list[AnyMessage], Blackboard], StageStateUpdate]
OnToolObservationCallback = Callable[[list[ToolObservationRecord], list[AnyMessage], Blackboard], StageStateUpdate]
OnInputCallback = Callable[[str, list[AnyMessage], Blackboard], StageStateUpdate]
OnOutputCallback = Callable[[str, Blackboard], None]
OnPlanningCallback = Callable[[Any, list[AnyMessage], Blackboard], StageStateUpdate]


def _apply_guard_incident_halt(updates: dict[str, Any]) -> None:
    if updates.get("guard_incident_halt"):
        updates["jump_to"] = "end"


_GUARD_PATCH_SKIP_KEYS = frozenset({"guard_regenerate_instruction", "messages"})


def _merge_guard_patch(updates: dict[str, Any], patch: dict[str, Any]) -> None:
    updates.update(
        {key: value for key, value in patch.items() if key not in _GUARD_PATCH_SKIP_KEYS}
    )


def _guard_patch_is_halt(patch: dict[str, Any]) -> bool:
    return bool(patch.get("guard_incident_halt"))


class MainStageCaptureMiddleware(AgentMiddleware[StageCaptureState, Any, Any]):
    """Expose tool selection (after_model) and tool observation (before_model) in state."""

    state_schema = StageCaptureState

    def __init__(
        self,
        *,
        debug: bool = False,
        on_tool_selection: OnToolSelectionCallback | None = None,
        on_tool_observation: OnToolObservationCallback | None = None,
        on_planning: OnPlanningCallback | None = None,
        recover_guidance: bool = True,
    ) -> None:
        self._debug = debug
        self._on_tool_selection = on_tool_selection
        self._on_tool_observation = on_tool_observation
        self._on_planning = on_planning
        self._recover_guidance = recover_guidance

    @hook_config(can_jump_to=["model", "end"])
    def after_model(
        self, state: StageCaptureState, runtime: Any  # noqa: ARG002
    ) -> dict[str, Any] | None:
        messages = list(state.get("messages") or [])
        pending = extract_pending_tool_selection(messages)
        if not pending:
            return None

        from attack_framework import try_apply_rts_tool_selection

        messages, pending, rts_event, rts_updates = try_apply_rts_tool_selection(
            state, messages, debug=self._debug
        )

        if self._debug:
            emit_stage_debug(STAGE_TOOL_SELECTION, stage_snapshot(STAGE_TOOL_SELECTION, pending)[STAGE_TOOL_SELECTION])

        updates: dict[str, Any] = {
            "last_tool_selection": pending,
            "blackboard": stage_snapshot(STAGE_TOOL_SELECTION, pending),
            **rts_updates,
        }
        prior = list(state.get("stage_events") or [])
        if rts_event is not None:
            prior = [*prior, rts_event]

        regenerate_instruction = ""
        guard_recover_applied = False
        message_deltas: list[AnyMessage] = []
        if self._on_tool_selection is not None:
            patch = self._on_tool_selection(
                pending, messages,
                merge_blackboard(state.get("blackboard"), updates["blackboard"]),
            )
            if patch:
                if patch.get("messages"):
                    message_deltas.extend(patch["messages"])
                if _guard_patch_is_halt(patch):
                    _merge_guard_patch(updates, patch)
                    pending = []
                    updates["last_tool_selection"] = []
                else:
                    guard_recover_applied = True
                    regenerate_instruction = str(
                        patch.get("guard_regenerate_instruction") or ""
                    )
                    _merge_guard_patch(updates, patch)
                    projected_messages = apply_message_deltas(messages, message_deltas)
                    recovered_pending = extract_pending_tool_selection(projected_messages)
                    if recovered_pending:
                        pending = recovered_pending
                        updates["last_tool_selection"] = pending
                    else:
                        pending, _, continuation_deltas = (
                            apply_tool_selection_recover_continuation(
                                projected_messages,
                                regenerate_instruction=regenerate_instruction,
                                inject_guidance=self._recover_guidance,
                            )
                        )
                        message_deltas.extend(continuation_deltas)
                        updates["last_tool_selection"] = pending

        # Recover cleared all pending tool calls — never enter ToolNode this step.
        if guard_recover_applied and not pending:
            updates["jump_to"] = "model"

        # planning: write_todos tool call args contain the natural-language plan todos
        if not updates.get("guard_incident_halt"):
            for call in pending:
                if call.get("name") == WRITE_TODOS_TOOL_NAME:
                    todos = (call.get("args") or {}).get("todos")
                    if todos is not None:
                        updates["blackboard"].update(
                            stage_snapshot(STAGE_PLANNING, todos)
                        )
                        plan_patch = (
                            self._on_planning(
                                todos, messages,
                                merge_blackboard(state.get("blackboard"), updates["blackboard"]),
                            )
                            if self._on_planning is not None
                            else None
                        )
                        if plan_patch:
                            if plan_patch.get("messages"):
                                message_deltas.extend(plan_patch["messages"])
                            if _guard_patch_is_halt(plan_patch):
                                _merge_guard_patch(updates, plan_patch)
                                pending = []
                                updates["last_tool_selection"] = []
                            else:
                                _merge_guard_patch(updates, plan_patch)
                                if self._recover_guidance:
                                    planning_regenerate = str(
                                        plan_patch.get("guard_regenerate_instruction")
                                        or ""
                                    )
                                    updates["guard_planning_recover_pending"] = True
                                    updates["guard_planning_recover_notice"] = (
                                        planning_regenerate
                                    )
                                projected_messages = apply_message_deltas(
                                    messages, message_deltas
                                )
                                pending = extract_pending_tool_selection(
                                    projected_messages
                                )
                                updates["last_tool_selection"] = pending
                    break

        if updates.get("guard_incident_halt"):
            pending = []
            updates["last_tool_selection"] = []
        _apply_guard_incident_halt(updates)

        if message_deltas:
            updates["messages"] = message_deltas

        event: StageEvent = {
            "stage": STAGE_TOOL_SELECTION,
            "tool_calls": stage_snapshot(STAGE_TOOL_SELECTION, pending)[STAGE_TOOL_SELECTION],
        }
        updates["stage_events"] = [*prior, event]
        return updates

    async def aafter_model(
        self, state: StageCaptureState, runtime: Any  # noqa: ARG002
    ) -> dict[str, Any] | None:
        return self.after_model(state, runtime)

    @hook_config(can_jump_to=["end"])
    def before_model(
        self, state: StageCaptureState, runtime: Any  # noqa: ARG002
    ) -> dict[str, Any] | None:
        messages = state.get("messages") or []
        updates: dict[str, Any] = {}

        if state.get("guard_planning_recover_pending") and is_completed_tool_step(messages):
            invocations = extract_last_step_invocations(messages)
            if any(inv.get("tool") == WRITE_TODOS_TOOL_NAME for inv in invocations):
                notice = build_planning_recover_notice(
                    regenerate_instruction=str(
                        state.get("guard_planning_recover_notice") or ""
                    )
                )
                updates["messages"] = [guard_notice_message(notice)]
                updates["guard_planning_recover_pending"] = False

        observations = extract_latest_tool_observations(messages)
        if not observations and not updates:
            return None

        if observations:
            if self._debug:
                emit_stage_debug(STAGE_TOOL_OBSERVATION, observations)

            updates["last_tool_observations"] = observations
            invocations = extract_last_step_invocations(messages)
            observation_texts = [record["content"] for record in observations]
            updates["blackboard"] = stage_snapshot(
                STAGE_TOOL_OBSERVATION,
                {"observation": observation_texts[0] if len(observation_texts) == 1
                 else observation_texts},
            )
            # Ordinary episodic searches also enter the memory stage; the MP
            # forced-search middleware captures its own pre-recovery snapshot.
            from episodic_memory import SEARCH_PAST_CONVERSATIONS_TOOL_NAME, STAGE_MEMORY

            # The forced search is a synthetic one-call round, already captured
            # in ForceEpisodicSearchMiddleware before any recovery.
            last_ai = _last_ai_message(messages)
            forced_search = bool(
                state.get("mp_retrieval_done")
                and last_ai
                and len(last_ai.tool_calls) == 1
                and str(last_ai.tool_calls[0].get("id", "")).startswith("mp_search_")
            )

            for invocation in invocations:
                if invocation["tool"] != SEARCH_PAST_CONVERSATIONS_TOOL_NAME:
                    continue
                if forced_search:
                    continue
                try:
                    retrieval = json.loads(invocation["observation"])
                except (TypeError, json.JSONDecodeError):
                    retrieval = None
                if not isinstance(retrieval, dict):
                    retrieval = {
                        **invocation["args"],
                        "observation": invocation["observation"],
                    }
                updates["blackboard"].update(stage_snapshot(
                    STAGE_MEMORY,
                    {**retrieval, "retrieval_tool": SEARCH_PAST_CONVERSATIONS_TOOL_NAME},
                ))
            if self._on_tool_observation is not None:
                patch = self._on_tool_observation(
                    observations, messages,
                    merge_blackboard(state.get("blackboard"), updates["blackboard"]),
                )
                if patch:
                    updates.update(patch)

            event: StageEvent = {
                "stage": STAGE_TOOL_OBSERVATION,
                "observations": observations,
            }
            prior = list(state.get("stage_events") or [])
            updates["stage_events"] = [*prior, event]

        _apply_guard_incident_halt(updates)
        return updates or None

    async def abefore_model(
        self, state: StageCaptureState, runtime: Any  # noqa: ARG002
    ) -> dict[str, Any] | None:
        return self.before_model(state, runtime)


def create_stage_capture_middleware(
    *,
    debug: bool = False,
    on_tool_selection: OnToolSelectionCallback | None = None,
    on_tool_observation: OnToolObservationCallback | None = None,
    on_planning: OnPlanningCallback | None = None,
    recover_guidance: bool = True,
) -> MainStageCaptureMiddleware:
    return MainStageCaptureMiddleware(
        debug=debug,
        on_tool_selection=on_tool_selection,
        on_tool_observation=on_tool_observation,
        on_planning=on_planning,
        recover_guidance=recover_guidance,
    )


class InputStageMiddleware(AgentMiddleware[StageCaptureState, Any, Any]):
    """Bind user input once at invocation entry, then refresh it each round."""

    state_schema = StageCaptureState

    def __init__(self, *, debug: bool = False, on_input: OnInputCallback | None = None) -> None:
        self._debug = debug
        self._on_input = on_input

    def before_agent(
        self, state: StageCaptureState, runtime: Any  # noqa: ARG002
    ) -> dict[str, Any] | None:
        # This hook runs once at graph entry, before internal tool/recovery loops.
        # The messages reducer assigns IDs before this hook. Seen IDs also stop
        # sanitized/replayed history from being mistaken for a new user request.
        seen = list(state.get("seen_user_input_ids") or [])
        seen_ids = set(seen)
        prior = state.get("user_input_binding")
        candidate = None
        for message in state.get("messages") or []:
            if not is_external_user_message(message):
                continue
            if not message.id:
                raise ValueError("User input binding requires a message ID at graph entry.")
            if message.id in seen_ids:
                # A real caller may edit the current user message in place.
                # Guard-sanitized copies carry internal origin and are excluded.
                if (prior and message.id == prior["message_id"]
                        and _tool_message_content(message.content) != prior["content"]):
                    candidate = message
                continue
            seen.append(message.id)
            seen_ids.add(message.id)
            candidate = message
        if candidate is None:
            return None
        updates: dict[str, Any] = {"seen_user_input_ids": seen}
        content = _tool_message_content(candidate.content)
        updates["user_input_binding"] = {
            "source": "user",
            "message_id": candidate.id,
            "version": (prior["version"] if prior else 0) + 1,
            "content": content,
        }
        return updates

    async def abefore_agent(
        self, state: StageCaptureState, runtime: Any  # noqa: ARG002
    ) -> dict[str, Any] | None:
        return self.before_agent(state, runtime)

    @hook_config(can_jump_to=["end"])
    def before_model(
        self, state: StageCaptureState, runtime: Any  # noqa: ARG002
    ) -> dict[str, Any] | None:
        messages = state.get("messages") or []
        binding = state.get("user_input_binding")
        user_input = binding["content"] if binding else None
        # Do not rescan model-facing messages: recovery notices and sanitized
        # copies cannot replace the request captured at the invocation boundary.
        if not user_input:
            return None
        if state.get("last_user_input") == user_input:
            # The round middleware archives prior snapshots before this hook.
            # Refresh the goal without repeating input checks or stage events.
            return {"blackboard": stage_snapshot(STAGE_INPUT, user_input)}

        if self._debug:
            emit_stage_debug(STAGE_INPUT, {"user_input": user_input})

        updates: dict[str, Any] = {
            "last_user_input": user_input,
            "blackboard": stage_snapshot(STAGE_INPUT, user_input),
        }
        if self._on_input is not None:
            patch = self._on_input(
                user_input, messages,
                merge_blackboard(state.get("blackboard"), updates["blackboard"]),
            )
            if patch:
                updates.update(patch)

        event: StageEvent = {
            "stage": STAGE_INPUT,
            "user_input": user_input,
        }
        prior = list(state.get("stage_events") or [])
        updates["stage_events"] = [*prior, event]
        _apply_guard_incident_halt(updates)
        return updates

    async def abefore_model(
        self, state: StageCaptureState, runtime: Any  # noqa: ARG002
    ) -> dict[str, Any] | None:
        return self.before_model(state, runtime)


def create_input_stage_middleware(
    *, debug: bool = False, on_input: OnInputCallback | None = None
) -> InputStageMiddleware:
    return InputStageMiddleware(debug=debug, on_input=on_input)


OnPostStepCallback = Callable[[dict[str, Any], list[AnyMessage], Blackboard], StageStateUpdate]


class PostStepStageMiddleware(AgentMiddleware[StageCaptureState, Any, Any]):
    """Fire after tool observation, before the next model call (end of one tool loop)."""

    state_schema = StageCaptureState

    def __init__(
        self,
        *,
        debug: bool = False,
        on_post_step: OnPostStepCallback | None = None,
    ) -> None:
        self._debug = debug
        self._on_post_step = on_post_step

    @hook_config(can_jump_to=["end"])
    def before_model(
        self, state: StageCaptureState, runtime: Any  # noqa: ARG002
    ) -> dict[str, Any] | None:
        messages = state.get("messages") or []
        if not is_completed_tool_step(messages):
            return None

        payload = build_post_step_payload(messages)

        if self._debug:
            emit_post_step_marker()
            emit_stage_debug(STAGE_POST_STEP, payload)

        updates: dict[str, Any] = {
            "blackboard": stage_snapshot(STAGE_POST_STEP, payload),
        }
        if self._on_post_step is not None:
            patch = self._on_post_step(
                payload, messages,
                merge_blackboard(state.get("blackboard"), updates["blackboard"]),
            )
            if patch:
                updates.update(patch)

        if updates.get("guard_incident_halt"):
            updates["jump_to"] = "end"

        event: StageEvent = {
            "stage": STAGE_POST_STEP,
            "invocations": payload["invocations"],
        }
        prior = list(state.get("stage_events") or [])
        updates["stage_events"] = [*prior, event]
        return updates or None

    async def abefore_model(
        self, state: StageCaptureState, runtime: Any  # noqa: ARG002
    ) -> dict[str, Any] | None:
        return self.before_model(state, runtime)


def create_post_step_middleware(
    *,
    debug: bool = False,
    on_post_step: OnPostStepCallback | None = None,
) -> PostStepStageMiddleware:
    return PostStepStageMiddleware(debug=debug, on_post_step=on_post_step)


class OutputStageMiddleware(AgentMiddleware[StageCaptureState, Any, Any]):
    """Capture final model output after each model call."""

    state_schema = StageCaptureState

    def __init__(self, *, debug: bool = False, on_output: OnOutputCallback | None = None) -> None:
        self._debug = debug
        self._on_output = on_output

    def after_model(
        self, state: StageCaptureState, runtime: Any  # noqa: ARG002
    ) -> dict[str, Any] | None:
        messages = state.get("messages") or []
        model_output = extract_final_model_output(messages)
        if not model_output:
            return None
        if state.get("last_model_output") == model_output:
            return None

        if self._debug:
            emit_stage_debug(STAGE_OUTPUT, {"model_output": model_output})

        if self._on_output is not None:
            self._on_output(
                model_output,
                merge_blackboard(state.get("blackboard"), stage_snapshot(STAGE_OUTPUT, model_output)),
            )

        event: StageEvent = {
            "stage": STAGE_OUTPUT,
            "model_output": model_output,
        }
        prior = list(state.get("stage_events") or [])
        return {
            "last_model_output": model_output,
            "blackboard": stage_snapshot(STAGE_OUTPUT, model_output),
            "stage_events": [*prior, event],
        }

    async def aafter_model(
        self, state: StageCaptureState, runtime: Any  # noqa: ARG002
    ) -> dict[str, Any] | None:
        return self.after_model(state, runtime)


def create_output_stage_middleware(
    *, debug: bool = False, on_output: OnOutputCallback | None = None
) -> OutputStageMiddleware:
    return OutputStageMiddleware(debug=debug, on_output=on_output)


class BlackboardRoundMiddleware(AgentMiddleware[StageCaptureState, Any, Any]):
    """Start the next round after observation/post_step hooks, before the model.

    Register after previous-step capture hooks and before input and forced memory
    retrieval. Archive the completed round, retain the previous two rounds, then
    refresh input for the current round.
    A recovery jump also starts a fresh round, even without tool execution.
    """

    state_schema = StageCaptureState

    def before_model(self, state: StageCaptureState, runtime: Any) -> dict[str, Any]:
        previous_round = state.get("blackboard_round", 0)
        # Ordinary merging would leak older stage keys into the new round.
        return {
            "blackboard_round": previous_round + 1,
            "blackboard": Overwrite(start_blackboard_round(
                state.get("blackboard"), previous_round=previous_round,
            )),
        }

    async def abefore_model(self, state: StageCaptureState, runtime: Any) -> dict[str, Any]:
        return self.before_model(state, runtime)

"""Capture Main Agent stage payloads for GuardAgent integration.

- before_model (input): latest user input before a model call
- after_model (tool selection): pending tool_calls before ToolNode runs
- before_model (tool observation): ToolMessage results before the next model call
- before_model (post step): after a completed tool loop, before the next model call
- after_model (output): final model output with no tool calls
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from typing import Any, NotRequired, TypedDict

from langchain.agents.middleware.types import AgentMiddleware, AgentState
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage

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


class StageCaptureState(AgentState):
    """Extended agent state for stage-aware guard hooks."""

    last_user_input: NotRequired[str]
    last_model_output: NotRequired[str]
    last_tool_selection: NotRequired[list[ToolCallPlan]]
    last_tool_observations: NotRequired[list[ToolObservationRecord]]
    stage_events: NotRequired[list[StageEvent]]
    guard_checks: NotRequired[dict[str, Any]]


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
        if isinstance(message, HumanMessage):
            return message
    return None


def extract_latest_user_input(messages: list[AnyMessage]) -> str | None:
    """Latest user input text before a model call."""
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


def emit_post_step_marker(*, stream: Any = sys.stderr) -> None:
    print("\n post step stage\n", file=stream)


def is_completed_tool_step(messages: list[AnyMessage]) -> bool:
    """True when the latest AIMessage's tool calls all have ToolMessage results."""
    return len(extract_latest_tool_observations(messages)) > 0


OnToolSelectionCallback = Callable[[list[ToolCallPlan]], None]
OnToolObservationCallback = Callable[[list[ToolObservationRecord]], None]
OnInputCallback = Callable[[str], None]
OnOutputCallback = Callable[[str], None]
OnPlanningCallback = Callable[[Any], None]


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
    ) -> None:
        self._debug = debug
        self._on_tool_selection = on_tool_selection
        self._on_tool_observation = on_tool_observation
        self._on_planning = on_planning

    def after_model(
        self, state: StageCaptureState, runtime: Any  # noqa: ARG002
    ) -> dict[str, Any] | None:
        messages = state.get("messages") or []
        pending = extract_pending_tool_selection(messages)
        if not pending:
            return None

        if self._debug:
            emit_stage_debug(STAGE_TOOL_SELECTION, pending)

        if self._on_tool_selection is not None:
            self._on_tool_selection(pending)

        # planning: write_todos tool call args contain the natural-language plan todos
        if self._on_planning is not None:
            for call in pending:
                if call.get("name") == WRITE_TODOS_TOOL_NAME:
                    todos = (call.get("args") or {}).get("todos")
                    if todos is not None:
                        self._on_planning(todos)
                    break

        event: StageEvent = {
            "stage": STAGE_TOOL_SELECTION,
            "tool_calls": pending,
        }
        prior = list(state.get("stage_events") or [])
        return {
            "last_tool_selection": pending,
            "stage_events": [*prior, event],
        }

    async def aafter_model(
        self, state: StageCaptureState, runtime: Any  # noqa: ARG002
    ) -> dict[str, Any] | None:
        return self.after_model(state, runtime)

    def before_model(
        self, state: StageCaptureState, runtime: Any  # noqa: ARG002
    ) -> dict[str, Any] | None:
        messages = state.get("messages") or []
        observations = extract_latest_tool_observations(messages)
        if not observations:
            return None

        if self._debug:
            emit_stage_debug(STAGE_TOOL_OBSERVATION, observations)

        if self._on_tool_observation is not None:
            self._on_tool_observation(observations)

        event: StageEvent = {
            "stage": STAGE_TOOL_OBSERVATION,
            "observations": observations,
        }
        prior = list(state.get("stage_events") or [])
        return {
            "last_tool_observations": observations,
            "stage_events": [*prior, event],
        }

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
) -> MainStageCaptureMiddleware:
    return MainStageCaptureMiddleware(
        debug=debug,
        on_tool_selection=on_tool_selection,
        on_tool_observation=on_tool_observation,
        on_planning=on_planning,
    )


class InputStageMiddleware(AgentMiddleware[StageCaptureState, Any, Any]):
    """Capture latest user input before each model call."""

    state_schema = StageCaptureState

    def __init__(self, *, debug: bool = False, on_input: OnInputCallback | None = None) -> None:
        self._debug = debug
        self._on_input = on_input

    def before_model(
        self, state: StageCaptureState, runtime: Any  # noqa: ARG002
    ) -> dict[str, Any] | None:
        messages = state.get("messages") or []
        user_input = extract_latest_user_input(messages)
        if not user_input:
            return None
        if state.get("last_user_input") == user_input:
            return None

        if self._debug:
            emit_stage_debug(STAGE_INPUT, {"user_input": user_input})

        if self._on_input is not None:
            self._on_input(user_input)

        event: StageEvent = {
            "stage": STAGE_INPUT,
            "user_input": user_input,
        }
        prior = list(state.get("stage_events") or [])
        return {
            "last_user_input": user_input,
            "stage_events": [*prior, event],
        }

    async def abefore_model(
        self, state: StageCaptureState, runtime: Any  # noqa: ARG002
    ) -> dict[str, Any] | None:
        return self.before_model(state, runtime)


def create_input_stage_middleware(
    *, debug: bool = False, on_input: OnInputCallback | None = None
) -> InputStageMiddleware:
    return InputStageMiddleware(debug=debug, on_input=on_input)


class PostStepStageMiddleware(AgentMiddleware[StageCaptureState, Any, Any]):
    """Fire after tool observation, before the next model call (end of one tool loop)."""

    state_schema = StageCaptureState

    def __init__(self, *, debug: bool = False) -> None:
        self._debug = debug

    def before_model(
        self, state: StageCaptureState, runtime: Any  # noqa: ARG002
    ) -> dict[str, Any] | None:
        messages = state.get("messages") or []
        if not is_completed_tool_step(messages):
            return None

        if self._debug:
            emit_post_step_marker()

        event: StageEvent = {"stage": STAGE_POST_STEP}
        prior = list(state.get("stage_events") or [])
        return {"stage_events": [*prior, event]}

    async def abefore_model(
        self, state: StageCaptureState, runtime: Any  # noqa: ARG002
    ) -> dict[str, Any] | None:
        return self.before_model(state, runtime)


def create_post_step_middleware(*, debug: bool = False) -> PostStepStageMiddleware:
    return PostStepStageMiddleware(debug=debug)


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
            self._on_output(model_output)

        event: StageEvent = {
            "stage": STAGE_OUTPUT,
            "model_output": model_output,
        }
        prior = list(state.get("stage_events") or [])
        return {
            "last_model_output": model_output,
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

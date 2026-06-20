"""Generic planning integration for DeepAgent (skill-agnostic).

References:
- https://docs.langchain.com/oss/python/deepagents/overview
- https://docs.langchain.com/oss/python/deepagents/deep-research
- https://docs.langchain.com/oss/python/langchain/middleware/built-in
- https://docs.langchain.com/oss/python/deepagents/customization

Domain-specific workflows belong in each skill's SKILL.md, not here.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware, TodoListMiddleware
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command
from deepagents import HarnessProfile, register_harness_profile

PLANNING_DEBUG_ENV = "DEEPAGENT_DEBUG_PLANNING"
PLANNING_REQUIRE_ENV = "DEEPAGENT_REQUIRE_PLANNING"
WRITE_TODOS_TOOL_NAME = "write_todos"

# Used when planning is not required: no mandatory write_todos workflow.
MINIMAL_SYSTEM_PROMPT = """\
You are a helpful agent. Use available tools when needed to complete the user's request.
You may use `write_todos` optionally for complex work, but you are **not** required to call it first.
"""

# USER slot: prepended before SDK BASE_AGENT_PROMPT (see create_deep_agent).
PLANNING_WORKFLOW_SYSTEM_PROMPT = """\
# Agent workflow

Follow this workflow for complex, multi-step requests:

1. **Plan first**: Call `write_todos` to break the work into focused, actionable steps **before any other tool**.
2. **Execute**: Work through todos in order. Mark each step `in_progress` before starting and `completed` immediately when done.
3. **Adapt**: Revise the todo list when new information changes the plan.
4. **Deliver**: After all todos are complete, provide the final answer in a normal assistant message.

## Planning rules

- For **non-trivial** tasks, `write_todos` MUST be your **first** tool call.
- Do **not** call any other tool until the initial todo list exists.
- Keep todos specific and outcome-oriented (e.g. "Load required inputs", "Run core analysis", "Produce final summary").
- For trivial one-step questions, skip the todo list and answer directly.

## Simple vs complex

| Complex — plan first | Simple — no todo list |
|----------------------|------------------------|
| 3+ distinct steps or tool calls | Single factual question |
| Multiple files, skills, or subtasks | Answer in one message |
| User gave multiple tasks | Pure conversation |
| Outcome may change as you learn more | No tools needed |
"""

# TodoListMiddleware system fragment (appended on each model call).
PLANNING_TODO_SYSTEM_PROMPT = """\
## `write_todos`

You have access to the `write_todos` tool to help you manage and plan complex objectives.

### Mandatory planning phase (non-trivial tasks)

For complex objectives (3+ steps, multiple tools, or multiple user tasks):

1. Your **first** action MUST be a single `write_todos` call that lists all steps.
2. Do **not** invoke any other tool in the same turn as that first `write_todos` call.
3. Only after the todo list exists may you call other tools.

Use this tool for complex objectives to ensure that you are tracking each necessary step.
This tool is very helpful for planning complex objectives, and for breaking down these larger complex objectives into smaller steps.

It is critical that you mark todos as completed as soon as you are done with a step. Do not batch up multiple steps before marking them as completed.
For simple objectives that only require a few steps, it is better to just complete the objective directly and NOT use this tool.
Writing todos takes time and tokens, use it when it is helpful for managing complex many-step problems! But not for simple few-step requests.

## Important To-Do List Usage Notes to Remember

- The `write_todos` tool should never be called multiple times in parallel.
- Don't be afraid to revise the To-Do list as you go. New information may reveal new tasks that need to be done, or old tasks that are irrelevant.

## Finishing a task

When you finish all work, write your final answer in the message AFTER your last `write_todos` call — not in the same turn as that call. Start the final message with the substantive content the user asked for — the data, computation, summary, or analysis. The user wants the result, not confirmation that the work is done.
"""

PLANNING_TODO_TOOL_DESCRIPTION = """\
Create and update a structured todo list for the current session.

**When required:** 3+ steps, multiple tools, or multiple user tasks — call this **first**, before any other tool. Mark the first item `in_progress`.

**When to skip:** trivial requests, fewer than 3 steps, or pure conversation — complete the task directly.

**Usage:** Mark items `in_progress` before work and `completed` when fully done; revise the list as you learn more. Do not call `write_todos` in parallel. Do not edit completed items.

**States:** `pending` | `in_progress` | `completed`

**Finish:** Deliver the user's answer in a later assistant message, not in the same turn as your last `write_todos` call.
"""


def planning_debug_enabled(cli_flag: bool = False) -> bool:
    import os

    if cli_flag:
        return True
    return os.environ.get(PLANNING_DEBUG_ENV, "").strip().lower() in ("1", "true", "yes")


def require_planning_enabled(cli_flag: bool = False) -> bool:
    """True when the agent must plan first via write_todos (CLI flag or env)."""
    import os

    if cli_flag:
        return True
    return os.environ.get(PLANNING_REQUIRE_ENV, "").strip().lower() in ("1", "true", "yes")


def format_plan_for_console(todos: Any) -> str:
    if isinstance(todos, list):
        return json.dumps(todos, ensure_ascii=False, indent=2)
    return json.dumps(todos, ensure_ascii=False, indent=2, default=str)


def emit_planning_debug(todos: Any, *, stream: Any = sys.stderr) -> None:
    print("\n agent planning...\n", file=stream)
    print(format_plan_for_console(todos), file=stream)
    print(file=stream)


def _state_todos(state: Any) -> list[Any] | None:
    if isinstance(state, dict):
        todos = state.get("todos")
        return todos if isinstance(todos, list) and todos else None
    return None


def _planning_required_message() -> str:
    return (
        "Error: Planning is required. Your **first** tool call must be `write_todos` "
        "with a step-by-step plan. Do not call other tools until the todo list exists."
    )


PLANNING_REQUIRED_ERROR_PREFIX = "Error: Planning is required."


def is_planning_required_tool_error(content: str) -> bool:
    return content.strip().startswith(PLANNING_REQUIRED_ERROR_PREFIX)


class RequirePlanningMiddleware(AgentMiddleware):
    """Block non-write_todos tools until the agent has created a todo list."""

    def after_model(
        self, state: Any, runtime: Any  # noqa: ARG002
    ) -> dict[str, Any] | None:
        if _state_todos(state) is not None:
            return None

        messages = state.get("messages") if isinstance(state, dict) else None
        if not messages:
            return None

        last_ai: AIMessage | None = None
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                last_ai = message
                break
        if last_ai is None or not last_ai.tool_calls:
            return None

        blocked: list[ToolMessage] = []
        for tool_call in last_ai.tool_calls:
            if tool_call.get("name") == WRITE_TODOS_TOOL_NAME:
                continue
            call_id = tool_call.get("id") or ""
            blocked.append(
                ToolMessage(
                    content=_planning_required_message(),
                    tool_call_id=call_id,
                    status="error",
                )
            )
        if not blocked:
            return None
        return {"messages": blocked}

    async def aafter_model(
        self, state: Any, runtime: Any  # noqa: ARG002
    ) -> dict[str, Any] | None:
        return self.after_model(state, runtime)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        name = request.tool_call.get("name")
        if name == WRITE_TODOS_TOOL_NAME:
            emit_planning_debug(request.tool_call.get("args", {}).get("todos", []))
            return handler(request)

        if _state_todos(request.state) is None:
            return ToolMessage(
                content=_planning_required_message(),
                tool_call_id=request.tool_call.get("id") or "",
                status="error",
            )
        return handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        name = request.tool_call.get("name")
        if name == WRITE_TODOS_TOOL_NAME:
            emit_planning_debug(request.tool_call.get("args", {}).get("todos", []))
            return await handler(request)

        if _state_todos(request.state) is None:
            return ToolMessage(
                content=_planning_required_message(),
                tool_call_id=request.tool_call.get("id") or "",
                status="error",
            )
        return await handler(request)


class PlanningDebugMiddleware(AgentMiddleware):
    """Log write_todos invocations to stderr when planning debug is enabled."""

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        if request.tool_call.get("name") == WRITE_TODOS_TOOL_NAME:
            emit_planning_debug(request.tool_call.get("args", {}).get("todos", []))
        return handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        if request.tool_call.get("name") == WRITE_TODOS_TOOL_NAME:
            emit_planning_debug(request.tool_call.get("args", {}).get("todos", []))
        return await handler(request)


class PlanningTodoListMiddleware(TodoListMiddleware):
    """Subclass so HarnessProfile can exclude default TodoListMiddleware only."""

    pass


def create_planning_todo_middleware() -> PlanningTodoListMiddleware:
    return PlanningTodoListMiddleware(
        system_prompt=PLANNING_TODO_SYSTEM_PROMPT,
        tool_description=PLANNING_TODO_TOOL_DESCRIPTION,
    )


def build_planning_middleware(
    *,
    debug_planning: bool = False,
    require_planning: bool = True,
) -> list[AgentMiddleware]:
    if not require_planning:
        return []
    stack: list[AgentMiddleware] = [
        create_planning_todo_middleware(),
        RequirePlanningMiddleware(),
    ]
    # When require_planning is on, RequirePlanningMiddleware already logs write_todos.
    if debug_planning and not require_planning:
        stack.append(PlanningDebugMiddleware())
    return stack


_PLANNING_PROFILE_OVERLAY = HarnessProfile(
    excluded_middleware=frozenset({TodoListMiddleware}),
)

_registered = False


def register_planning_harness_profile(*, require_planning: bool = True) -> None:
    """Merge planning profile overlay onto provider keys used by this demo."""
    global _registered
    if not require_planning:
        return
    if _registered:
        return
    for provider in ("openai", "deepseek"):
        register_harness_profile(provider, _PLANNING_PROFILE_OVERLAY)
    _registered = True

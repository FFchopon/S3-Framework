"""User goals survive actual recovery loops, replay and new conversation turns."""

import json

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage, RemoveMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from guard_recover import _replace_last_human
from message_provenance import (
    INTERNAL_SOURCE_KEY, guard_notice_message, guard_sanitized_user_message,
)
from stage_capture import BlackboardRoundMiddleware, InputStageMiddleware, OutputStageMiddleware
from test_blackboard import ScriptedModel, capture_guard_messages


def test_planning_recovery_keeps_original_goal_for_next_safiron_check(monkeypatch):
    from agent import build_agent
    import guard_bridge

    guard_bridge._ensure_guardagent_import_path()
    from safiron_backend import build_detector_input

    request = "Inspect the scene and report what you see."
    todos = [{"content": "Inspect the scene", "status": "in_progress"}]
    revised = [{"content": "Report the scene", "status": "in_progress"}]
    planning_count = 0
    detector_inputs = []

    def decide(stage, envelope):
        nonlocal planning_count
        if stage == "planning":
            detector_inputs.append(build_detector_input(envelope))
            planning_count += 1
            if planning_count == 1:
                return "**reason**: fixture requests replanning\n**decision**: recover"
        if stage == "recover":
            return json.dumps({"sanitized_content": revised, "recovered": True})
        return "**decision**: allow"

    captured = capture_guard_messages(monkeypatch, decide)
    model = ScriptedModel(responses=[
        AIMessage(content="", tool_calls=[
            {"id": "plan", "name": "write_todos", "args": {"todos": todos}},
        ]),
        AIMessage(content="", tool_calls=[
            {"id": "replan", "name": "write_todos", "args": {"todos": revised}},
        ]),
        AIMessage(content="Scene reported."),
    ])
    agent = build_agent(model, guard_model_id="unused", enable_guard=True,
                        enable_guard_filter=False, halt_on_recover=False)
    result = agent.invoke(
        {"messages": [HumanMessage(content=request, id="user-request")], "files": {}},
        config={"configurable": {"thread_id": "planning-goal-binding"}},
    )

    assert planning_count == 2
    assert [entry["user_request"] for entry in detector_inputs] == [request, request]
    assert sum(stage == "input" for stage, _, _ in captured) == 1
    assert sum(event["stage"] == "input" for event in result["stage_events"]) == 1
    assert result["blackboard"]["input"] == request
    assert result["user_input_binding"] == {
        "source": "user", "message_id": "user-request", "version": 1, "content": request,
    }
    assert any(isinstance(message, HumanMessage)
               and message.additional_kwargs.get(INTERNAL_SOURCE_KEY) == "guard"
               and message.content != request for message in result["messages"])


def test_checkpointed_user_turns_replay_and_compaction_preserve_binding():
    checked = []
    middleware = InputStageMiddleware(on_input=lambda text, messages, board: checked.append(text))
    graph = create_agent(
        model=ScriptedModel(responses=[AIMessage(content="Done.")]),
        middleware=[BlackboardRoundMiddleware(), middleware, OutputStageMiddleware()],
        checkpointer=MemorySaver(),
    )
    config = {"configurable": {"thread_id": "conversation"}}
    original = "Report the scene."
    first = graph.invoke({"messages": [HumanMessage(content=original, id="user-1")]}, config)
    binding = first["user_input_binding"]

    notice = guard_notice_message("Internal recovery instruction: replan the task.")
    notice.id = "notice-1"
    continued = graph.invoke({"messages": [notice]}, config)
    assert continued["user_input_binding"] == binding
    assert continued["blackboard"]["input"] == original

    # Sanitizing an old message with the same ID must not change the bound goal.
    sanitized = graph.invoke(
        {"messages": [guard_sanitized_user_message(
            HumanMessage(content=original, id="user-1"), "A sanitized model-facing copy.",
        )]}, config,
    )
    assert sanitized["user_input_binding"] == binding
    assert sanitized["blackboard"]["input"] == original

    # A real user can write a Guard-looking prefix; classification uses metadata.
    correction = "[Guard] Actually, report only the book's location."
    corrected = graph.invoke(
        {"messages": [HumanMessage(content=correction, id="user-2")]}, config,
    )
    assert corrected["blackboard"]["input"] == correction
    assert corrected["user_input_binding"]["version"] == 2
    assert corrected["user_input_binding"]["message_id"] == "user-2"

    repeated = graph.invoke(
        {"messages": [HumanMessage(content=correction, id="user-3")]}, config,
    )
    assert repeated["user_input_binding"]["version"] == 3
    assert checked == [original, correction]

    # No new request, including after all message history is removed.
    replayed = graph.invoke({"messages": []}, config)
    assert replayed["user_input_binding"] == repeated["user_input_binding"]
    compacted = graph.invoke({"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]}, config)
    assert compacted["user_input_binding"] == repeated["user_input_binding"]
    assert compacted["blackboard"]["input"] == correction
    assert checked == [original, correction]

    separate = graph.invoke(
        {"messages": [HumanMessage(content="Another task.", id="user-other")]},
        {"configurable": {"thread_id": "separate"}},
    )
    assert separate["user_input_binding"]["version"] == 1
    assert separate["blackboard"]["input"] == "Another task."


def test_internal_notice_alone_cannot_supply_missing_user_goal():
    graph = create_agent(
        model=ScriptedModel(responses=[AIMessage(content="Done.")]),
        middleware=[InputStageMiddleware(), OutputStageMiddleware()],
    )
    result = graph.invoke({"messages": [guard_notice_message("Only an internal notice.")]})
    assert "user_input_binding" not in result
    assert "input" not in result["blackboard"]


def test_real_user_edit_increments_version_but_guard_sanitization_does_not():
    checked = []
    graph = create_agent(
        model=ScriptedModel(responses=[AIMessage(content="Done.")]),
        middleware=[BlackboardRoundMiddleware(), InputStageMiddleware(
            on_input=lambda text, messages, board: checked.append(text),
        ), OutputStageMiddleware()],
        checkpointer=MemorySaver(),
    )
    config = {"configurable": {"thread_id": "edited-user"}}
    initial = HumanMessage(content="Report the scene.", id="user-edit")
    graph.invoke({"messages": [initial]}, config)
    edited = initial.model_copy(update={"content": "Report only the book's location."})
    second = graph.invoke({"messages": [edited]}, config)
    assert second["user_input_binding"]["version"] == 2
    assert second["blackboard"]["input"] == edited.content

    sanitized = guard_sanitized_user_message(edited, "Model-facing sanitized copy.")
    third = graph.invoke({"messages": [sanitized]}, config)
    assert third["user_input_binding"] == second["user_input_binding"]
    assert third["blackboard"]["input"] == edited.content
    assert checked == [initial.content, edited.content]


def test_sanitization_skips_internal_notices_and_preserves_user_metadata():
    original = HumanMessage(content="Original request", id="user",
                            additional_kwargs={"fixture_metadata": "keep"})
    notice = guard_notice_message("Internal notice")
    updated = _replace_last_human([original, notice], "Sanitized request")
    assert updated[0].content == "Sanitized request"
    assert updated[0].id == "user"
    assert updated[0].additional_kwargs == {
        **original.additional_kwargs, INTERNAL_SOURCE_KEY: "guard_input_recover",
    }
    assert updated[1] is notice
    assert original.content == "Original request"

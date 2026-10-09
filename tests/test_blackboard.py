"""Offline blackboard integration: real stage hooks, tools, and graph state."""

import json
from types import SimpleNamespace

import pytest

from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

from blackboard import merge_blackboard, stage_snapshot
from episode_store import Episode, EpisodeRegistry
from episodic_memory import ForceEpisodicSearchMiddleware
from stage_capture import (
    BlackboardRoundMiddleware,
    InputStageMiddleware,
    MainStageCaptureMiddleware,
    OutputStageMiddleware,
    PostStepStageMiddleware,
)


class ScriptedModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def current_stages(board):
    return {key: value for key, value in board.items() if key not in ("round", "history")}


def test_stage_updates_preserve_other_stages_and_copy_nested_values():
    todos = [{"content": "Inspect book", "status": "pending"}]
    update = stage_snapshot("planning", todos)
    todos[0]["content"] = "changed after capture"
    first = merge_blackboard({"input": "request"}, update)
    update["planning"][0]["status"] = "completed"
    second = merge_blackboard(first, stage_snapshot("input", "new request"))
    second["planning"][0]["status"] = "in_progress"
    assert first == {
        "input": "request",
        "planning": [{"content": "Inspect book", "status": "pending"}],
    }
    assert second["input"] == "new request"


def test_graph_captures_all_stages_without_guard_and_isolates_threads():
    registry = EpisodeRegistry()
    registry.add_episode(Episode(
        episode_id="episode", user_id="task-one", title="historical request",
        user_input="Find the book", generated_plan=[{"content": "Find book"}],
        messages=[], is_poison=True, kind="risk",
    ))
    todos = [{"content": "Find the book", "status": "in_progress"}]
    find_call = {"id": "find-1", "name": "find", "args": {"item": "book"}}
    model = ScriptedModel(responses=[
        AIMessage(content="", tool_calls=[
            {"id": "plan-1", "name": "write_todos", "args": {"todos": todos}},
        ]),
        AIMessage(content="", tool_calls=[find_call]),
        AIMessage(content="The book is on the shelf."),
        AIMessage(content="Second task finished."),
    ])

    @tool
    def find(item: str) -> str:
        """Locate an item in the fixture scene."""
        return f"{item} is on the shelf"

    agent = create_agent(
        model=model,
        tools=[find],
        middleware=[
            TodoListMiddleware(), InputStageMiddleware(),
            ForceEpisodicSearchMiddleware(registry=registry),
            MainStageCaptureMiddleware(), PostStepStageMiddleware(),
            OutputStageMiddleware(),
        ],
        checkpointer=MemorySaver(),
    )
    first = agent.invoke(
        {"messages": [HumanMessage(content="Find the book")],
         "attack_type": "mp", "mp_user_id": "task-one"},
        config={"configurable": {"thread_id": "first"}},
    )
    board = first["blackboard"]
    assert set(board) == {
        "input", "memory", "planning", "tool_selection",
        "tool_observation", "post_step", "output",
    }
    assert board["input"] == "Find the book"
    assert board["planning"] == todos
    assert board["tool_selection"] == [{"name": "find", "args": {"item": "book"}}]
    assert board["tool_observation"] == {"observation": "book is on the shelf"}
    assert board["post_step"] == {
        "invocations": [{"tool": "find", "args": {"item": "book"},
                         "observation": "book is on the shelf"}],
    }
    assert board["output"] == "The book is on the shelf."
    assert set(board["memory"]["episodes"][0]) == {
        "rank", "user_input", "generated_plan",
    }
    # The blackboard never enters the conversation or adds a tool call.
    assert all("blackboard" not in str(message.content) for message in first["messages"])
    assert [message.name for message in first["messages"] if isinstance(message, ToolMessage)] == [
        "search_past_conversations", "write_todos", "find",
    ]
    assert [message.tool_call_id for message in first["messages"]
            if isinstance(message, ToolMessage) and message.name == "find"] == ["find-1"]
    assert all("id" not in call for event in first["stage_events"]
               if event["stage"] == "tool_selection" for call in event["tool_calls"])
    json.dumps(board)

    second = agent.invoke(
        {"messages": [HumanMessage(content="Another request")]},
        config={"configurable": {"thread_id": "second"}},
    )
    assert second["blackboard"] == {
        "input": "Another request", "output": "Second task finished.",
    }
    assert agent.get_state({"configurable": {"thread_id": "first"}}).values["blackboard"] == board


@pytest.mark.parametrize("contents", [
    ["fork is at cabinet."],
    [""],
    ["fork is at cabinet.", "book is at table."],
    ["same result", "same result"],
])
def test_observation_snapshot_keeps_only_complete_ordered_results(contents):
    calls = [{"id": f"read-{number}", "name": "find", "args": {"item": f"item-{number}"}}
             for number in range(len(contents))]
    messages = [AIMessage(content="", tool_calls=calls)] + [
        ToolMessage(content=content, tool_call_id=call["id"], name=call["name"])
        for call, content in zip(calls, contents)
    ]
    expected = contents[0] if len(contents) == 1 else contents
    received = []

    def inspect_observations(records, messages, board):
        received.extend(dict(record) for record in records)
        assert board["tool_observation"] == {"observation": expected}
        records[0]["content"] = "changed after capture"

    update = MainStageCaptureMiddleware(
        on_tool_observation=inspect_observations,
    ).before_model({"messages": messages}, None)
    assert update["blackboard"]["tool_observation"] == {"observation": expected}
    assert [record["content"] for record in received] == contents
    assert [record["tool_call_id"] for record in received] == [call["id"] for call in calls]
    post_step = PostStepStageMiddleware().before_model({"messages": messages}, None)
    assert post_step["blackboard"]["post_step"] == {"invocations": [
        {"tool": call["name"], "args": call["args"], "observation": content}
        for call, content in zip(calls, contents)
    ]}


def test_ordinary_search_captures_memory_with_tool_context():
    retrieval = {"query": "find book", "user_id": "task", "top_k": 4,
                 "episodes": [{"rank": 1, "user_input": "past", "generated_plan": []}]}
    messages = [
        AIMessage(content="", tool_calls=[
            {"id": "search-1", "name": "search_past_conversations",
             "args": {"query": "find book", "top_k": 4}},
        ]),
        ToolMessage(content=json.dumps(retrieval), tool_call_id="search-1",
                    name="search_past_conversations"),
    ]
    update = MainStageCaptureMiddleware().before_model({"messages": messages}, None)
    assert update["blackboard"]["memory"] == {
        **retrieval, "retrieval_tool": "search_past_conversations",
    }


def test_capture_survives_guard_halt_and_mutation():
    def halt_and_mutate(calls, messages, blackboard):
        calls[0]["args"]["item"] = "changed by recovery"
        return {"guard_incident_halt": True}

    update = MainStageCaptureMiddleware(
        on_tool_selection=halt_and_mutate,
    ).after_model({"messages": [AIMessage(content="", tool_calls=[
        {"id": "call", "name": "find", "args": {"item": "book"}},
    ])]}, None)
    assert update["jump_to"] == "end"
    assert update["last_tool_selection"] == []
    assert update["blackboard"]["tool_selection"][0]["args"] == {"item": "book"}

    registry = EpisodeRegistry()
    retrieval = ForceEpisodicSearchMiddleware(
        registry=registry,
        on_memory_retrieval=lambda *args: {"guard_incident_halt": True},
    ).before_model({
        "messages": [HumanMessage(content="request")],
        "attack_type": "mp", "mp_user_id": "user",
    }, None)
    assert retrieval["jump_to"] == "end"
    assert retrieval["blackboard"]["memory"]["query"] == "request"


def test_forced_memory_snapshot_is_not_replaced_by_sanitized_observation():
    messages = [
        AIMessage(content="", tool_calls=[
            {"id": "mp_search_fixture", "name": "search_past_conversations", "args": {}},
        ]),
        ToolMessage(content='{"episodes": []}', tool_call_id="mp_search_fixture",
                    name="search_past_conversations"),
    ]
    update = MainStageCaptureMiddleware().before_model({
        "messages": messages, "mp_retrieval_done": True,
    }, None)
    assert "memory" not in update["blackboard"]
    assert "tool_observation" in update["blackboard"]


def test_production_agent_writes_blackboard_without_guard():
    from agent import build_agent

    # This runs the actual Deep Agents assembly with a local scripted model.
    model = ScriptedModel(responses=[AIMessage(content="Completed offline.")])
    agent = build_agent(model, enable_guard=False)
    result = agent.invoke(
        {"messages": [HumanMessage(content="A local fixture request")], "files": {}},
        config={"configurable": {"thread_id": "blackboard-production-fixture"}},
    )
    assert current_stages(result["blackboard"]) == {
        "input": "A local fixture request", "output": "Completed offline.",
    }
    assert result["blackboard"]["round"] == 1
    assert result["blackboard"]["history"] == []


def capture_guard_messages(monkeypatch, decide=None):
    """Stub only the model invocation; exercise the production Guard bridge."""
    import guard_bridge

    guard_bridge._ensure_guardagent_import_path()
    from runtime import parse_guard_message

    captured = []

    def invoke(self, stage, message):
        user_message, world = parse_guard_message(message)
        envelope = json.loads(user_message)
        assert envelope["stage"] == stage
        assert stage not in envelope["blackboard"]
        captured.append((stage, envelope, world))
        content = decide(stage, envelope) if decide else "**decision**: allow"
        return SimpleNamespace(returncode=0, content=content, stdout="", stderr="",
                               embodied_world=None)

    monkeypatch.setattr(guard_bridge, "stage_has_guard_skill", lambda stage: True)
    monkeypatch.setattr(guard_bridge.GuardAgentClient, "_invoke", invoke)
    return captured


def test_production_guard_receives_round_context_for_all_stages(monkeypatch):
    import agent as agent_module
    from embodied_env import tools

    registry = EpisodeRegistry()
    registry.add_episode(Episode(
        episode_id="episode", user_id="fixture", title="past request",
        user_input="Inspect scene", generated_plan=[], messages=[],
    ))
    monkeypatch.setattr(agent_module, "EPISODE_REGISTRY", registry)
    captured = capture_guard_messages(monkeypatch)
    todos = [{"content": "Inspect scene", "status": "in_progress"}]
    model = ScriptedModel(responses=[
        AIMessage(content="", tool_calls=[
            {"id": "plan", "name": "write_todos", "args": {"todos": todos}},
        ]),
        AIMessage(content="", tool_calls=[
            {"id": "observe", "name": "observe_environment", "args": {}},
        ]),
        AIMessage(content="Scene inspected."),
    ])
    previous_world = tools.get_embodied_world_snapshot()
    try:
        agent = agent_module.build_agent(
            model, guard_model_id="unused", enable_guard=True,
            enable_guard_filter=False, embodied=True, attack="mp",
        )
        result = agent.invoke(
            {"messages": [HumanMessage(content="Inspect scene")], "files": {},
             "attack_type": "mp", "mp_user_id": "fixture"},
            config={"configurable": {"thread_id": "guard-rounds"}},
        )
    finally:
        tools.apply_embodied_world_snapshot(previous_world)

    by_stage = {}
    for stage, envelope, world in captured:
        by_stage.setdefault(stage, []).append(envelope)
        assert "embodied_world" not in envelope
        assert (world is not None) == (stage == "post_step")
    assert set(by_stage) == {
        "input", "memory", "planning", "tool_selection",
        "tool_observation", "post_step", "output",
    }
    assert by_stage["input"][0] == {
        "stage": "input", "stage_payload": "Inspect scene",
        "blackboard": {"round": 1, "history": []},
    }
    assert current_stages(by_stage["memory"][0]["blackboard"]) == {"input": "Inspect scene"}
    assert by_stage["planning"][0]["stage_payload"] == todos
    assert by_stage["planning"][0]["blackboard"]["input"] == "Inspect scene"
    assert "memory" in by_stage["tool_selection"][0]["blackboard"]
    # Plan and input survive through first round's post_step check.
    plan_post_step = next(
        item for item in by_stage["post_step"]
        if item["stage_payload"]["invocations"][0]["tool"] == "write_todos"
    )
    assert plan_post_step["blackboard"]["planning"] == todos
    assert plan_post_step["blackboard"]["input"] == "Inspect scene"
    # Only prior post-step evidence is archived; global input remains available.
    selection_board = by_stage["tool_selection"][1]["blackboard"]
    assert current_stages(selection_board) == {"input": "Inspect scene"}
    assert selection_board["history"][0]["round"] == 1
    assert set(selection_board["history"][0]["stages"]) == {"post_step"}
    assert selection_board["history"][0]["stages"]["post_step"]["invocations"][0]["args"]["todos"] == todos
    assert set(current_stages(by_stage["tool_observation"][0]["blackboard"])) == {"input", "tool_selection"}
    assert set(current_stages(by_stage["post_step"][-1]["blackboard"])) == {
        "input", "tool_selection", "tool_observation",
    }
    assert current_stages(by_stage["output"][0]["blackboard"]) == {"input": "Inspect scene"}
    assert current_stages(result["blackboard"]) == {"input": "Inspect scene", "output": "Scene inspected."}
    assert [item["round"] for item in result["blackboard"]["history"]] == [1, 2]
    assert result["blackboard_round"] == 3
    assert all("blackboard" not in str(message.content) for message in result["messages"])


def test_later_round_planning_receives_goal_without_rechecking_input(monkeypatch):
    from agent import build_agent
    from embodied_env import tools

    captured = capture_guard_messages(monkeypatch)
    request = "Inspect the scene and report what you see."
    plan = [{"content": "Report the observed scene", "status": "in_progress"}]
    completed = [{"content": "Report the observed scene", "status": "completed"}]
    model = ScriptedModel(responses=[
        AIMessage(content="", tool_calls=[
            {"id": "observe", "name": "observe_environment", "args": {}},
        ]),
        AIMessage(content="", tool_calls=[
            {"id": "plan", "name": "write_todos", "args": {"todos": plan}},
        ]),
        AIMessage(content="", tool_calls=[
            {"id": "complete", "name": "write_todos", "args": {"todos": completed}},
        ]),
        AIMessage(content="Scene inspected."),
    ])
    previous_world = tools.get_embodied_world_snapshot()
    try:
        agent = build_agent(
            model, guard_model_id="unused", enable_guard=True,
            enable_guard_filter=False, embodied=True,
        )
        result = agent.invoke(
            {"messages": [HumanMessage(content=request)], "files": {}},
            config={"configurable": {"thread_id": "later-round-goal"}},
        )
    finally:
        tools.apply_embodied_world_snapshot(previous_world)

    planning = [envelope for stage, envelope, _ in captured if stage == "planning"]
    assert [item["stage_payload"] for item in planning] == [plan, completed]
    for item in planning:
        assert item["blackboard"]["input"] == request
        assert "tool_observation" not in item["blackboard"]
        assert "post_step" not in item["blackboard"]
        assert item["blackboard"]["history"][0]["round"] == 1
        prior_stages = item["blackboard"]["history"][0]["stages"]
        assert set(prior_stages) == {"post_step"}
        assert prior_stages["post_step"]["invocations"][0]["tool"] == "observe_environment"
    assert sum(stage == "input" for stage, _, _ in captured) == 1
    assert sum(event["stage"] == "input" for event in result["stage_events"]) == 1
    assert current_stages(result["blackboard"]) == {"input": request, "output": "Scene inspected."}
    assert [item["round"] for item in result["blackboard"]["history"]] == [2, 3]
    assert result["blackboard_round"] == 4


@pytest.mark.parametrize("halt", [False, True])
def test_recovery_skip_or_halt_preserves_round_boundary(monkeypatch, halt):
    from agent import build_agent

    def decide(stage, envelope):
        if stage == "tool_selection":
            return "**decision**: recover\n**reason**: unsafe call"
        if stage == "recover":
            return json.dumps({"sanitized_content": [], "recovered": True})
        return "**decision**: allow"

    captured = capture_guard_messages(monkeypatch, decide)
    model = ScriptedModel(responses=[
        AIMessage(content="", tool_calls=[
            {"id": "blocked", "name": "write_todos", "args": {"todos": []}},
        ]),
        AIMessage(content="Completed safely."),
    ])
    agent = build_agent(
        model, guard_model_id="unused", enable_guard=True,
        enable_guard_filter=False, halt_on_recover=halt,
    )
    result = agent.invoke(
        {"messages": [HumanMessage(content="Benign request")], "files": {}},
        config={"configurable": {"thread_id": f"recovery-{halt}"}},
    )
    if halt:
        assert current_stages(result["blackboard"]) == {
            "input": "Benign request",
            "tool_selection": [{"name": "write_todos",
                                "args": {"todos": []}}],
        }
        assert result["blackboard_round"] == 1
        assert not any(stage == "recover" for stage, _, _ in captured)
    else:
        assert current_stages(result["blackboard"]) == {
            "input": "Benign request", "output": "Completed safely.",
        }
        assert result["blackboard_round"] == 2
        recovery = next(envelope for stage, envelope, _ in captured if stage == "recover")
        assert current_stages(recovery["blackboard"]) == {"input": "Benign request"}
        assert recovery["stage_payload"]["source_stage"] == "tool_selection"
        assert current_stages(captured[-1][1]["blackboard"]) == {"input": "Benign request"}
        assert result["blackboard"]["history"][0]["round"] == 1
        assert result["blackboard"]["history"][0]["stages"] == {}
        assert result["last_user_input"] == "Benign request"
        assert sum(stage == "input" for stage, _, _ in captured) == 1
    assert not any(isinstance(message, ToolMessage) for message in result["messages"])


def test_async_graph_retains_round_history_and_preserves_thread_isolation():
    import asyncio

    captured = []
    def on_selection(payload, messages, board):
        captured.append(board)

    model = ScriptedModel(responses=[
        AIMessage(content="", tool_calls=[
            {"id": "plan", "name": "write_todos", "args": {"todos": []}},
        ]),
        AIMessage(content="First finished."),
        AIMessage(content="Second finished."),
    ])
    agent = create_agent(
        model=model,
        middleware=[TodoListMiddleware(),
                    MainStageCaptureMiddleware(on_tool_selection=on_selection),
                    PostStepStageMiddleware(), OutputStageMiddleware(),
                    BlackboardRoundMiddleware(), InputStageMiddleware()],
        checkpointer=MemorySaver(),
    )
    async def run():
        first = await agent.ainvoke(
            {"messages": [HumanMessage(content="First request")]},
            config={"configurable": {"thread_id": "async-first"}},
        )
        second = await agent.ainvoke(
            {"messages": [HumanMessage(content="Second request")]},
            config={"configurable": {"thread_id": "async-second"}},
        )
        return first, second

    async def bounded_run():
        return await asyncio.wait_for(run(), timeout=15)

    first, second = asyncio.run(bounded_run())
    assert captured[0]["input"] == "First request"
    assert current_stages(first["blackboard"]) == {"input": "First request", "output": "First finished."}
    assert [item["round"] for item in first["blackboard"]["history"]] == [1]
    assert first["blackboard_round"] == 2
    assert current_stages(second["blackboard"]) == {"input": "Second request", "output": "Second finished."}
    assert second["blackboard"]["history"] == []
    assert second["blackboard_round"] == 1


@pytest.mark.parametrize("attack", ["dpi", "rte"])
def test_batch_saves_blackboard_without_guard(tmp_path, attack):
    from agent import run_embodied_batch
    from embodied_env import tools
    from result_writer import RunResultWriter, load_run_result

    class LocalAgent:
        def invoke(self, state, config):
            request = state["messages"][0]["content"]
            return {
                "messages": [AIMessage(content="Offline fixture.")],
                "blackboard": {"input": request, "output": "Offline fixture."},
            }

    previous_world = tools.get_embodied_world_snapshot()
    writer = RunResultWriter(tmp_path / "run.json", run={})
    try:
        run_embodied_batch(
            LocalAgent(), num=1, attack=attack, files={}, result_writer=writer,
        )
    finally:
        tools.apply_embodied_world_snapshot(previous_world)
    record = load_run_result(writer.path)["tasks"][0]
    board = record["blackboard"]
    assert board["input"] == record["user_message"]
    if attack == "rte":
        assert set(board) == {"round", "history", "input", "post_step"}
        assert board["round"] == 1 and board["history"] == []
        assert board["post_step"]["invocations"]
    else:
        assert board["output"] == "Offline fixture."

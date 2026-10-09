"""Three-round retention through real graph hooks and the Safiron input adapter."""

from copy import deepcopy

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from blackboard import guard_user_payload, start_blackboard_round
from stage_capture import (
    BlackboardRoundMiddleware, InputStageMiddleware, MainStageCaptureMiddleware,
    OutputStageMiddleware, PostStepStageMiddleware,
)
from test_blackboard import ScriptedModel


def test_three_round_window_keeps_distinct_observations_and_evicts_old_rounds():
    selection_envelopes = []

    @tool
    def inspect_fixture(sequence: int) -> str:
        """Return a distinct local fixture observation for this step."""
        return f"fixture-state-{sequence}"

    def inspect_selection(calls, messages, board):
        selection_envelopes.append(guard_user_payload("tool_selection", calls, board))

    responses = [AIMessage(content="", tool_calls=[
        {"id": f"read-{number}", "name": "inspect_fixture", "args": {"sequence": number}},
    ]) for number in range(1, 5)] + [AIMessage(content="Read four fixture states.")]
    graph = create_agent(
        model=ScriptedModel(responses=responses), tools=[inspect_fixture],
        middleware=[MainStageCaptureMiddleware(on_tool_selection=inspect_selection),
                    PostStepStageMiddleware(), OutputStageMiddleware(),
                    BlackboardRoundMiddleware(), InputStageMiddleware()],
    )
    result = graph.invoke({"messages": [HumanMessage(content="Inspect the fixture four times.")]})

    for number, envelope in enumerate(selection_envelopes, 1):
        board = envelope["blackboard"]
        assert board["round"] == number
        assert [entry["round"] for entry in board["history"]] == list(range(max(1, number - 2), number))
        assert "tool_selection" not in board
        assert "tool_observation" not in board
        for entry in board["history"]:
            stages = entry["stages"]
            assert set(stages) == {"post_step"}
            assert stages["post_step"]["invocations"][0]["args"]["sequence"] == entry["round"]
            expected_observation = f"fixture-state-{entry['round']}"
            assert stages["post_step"]["invocations"][0]["observation"] == expected_observation

    board = result["blackboard"]
    assert board["round"] == result["blackboard_round"] == 5
    assert [entry["round"] for entry in board["history"]] == [3, 4]
    assert board["input"] == "Inspect the fixture four times."
    assert board["output"] == "Read four fixture states."
    assert "tool_observation" not in board
    # Mutating a Guard view cannot alter a retained round or final graph state.
    selection_envelopes[-1]["blackboard"]["history"][-1]["stages"]["post_step"]["invocations"][0]["observation"] = "changed"
    assert board["history"][0]["stages"]["post_step"]["invocations"][0]["observation"] == "fixture-state-3"


def test_guard_excludes_only_current_stage_and_recovery_source_from_current_round():
    plan = [{"content": "Report the scene", "status": "pending"}]
    post_step = {"invocations": [{"tool": "find", "args": {}, "observation": "found"}]}
    board = {"round": 3, "input": "Report the scene.", "planning": plan,
             "recover": "current recovery",
             "history": [{"round": 1, "stages": {"post_step": post_step}},
                         {"round": 2, "stages": {"post_step": post_step}}]}
    original = deepcopy(board)
    planning = guard_user_payload("planning", plan, board)
    assert "planning" not in planning["blackboard"]
    assert planning["blackboard"]["history"][0]["stages"] == {"post_step": post_step}
    recovered = guard_user_payload("recover", {}, board, exclude_stages=("planning",))
    assert "planning" not in recovered["blackboard"] and "recover" not in recovered["blackboard"]
    assert recovered["blackboard"]["history"] == original["history"]
    planning["blackboard"]["history"][0]["stages"]["post_step"]["invocations"][0]["observation"] = "changed"
    assert board == original


def test_safiron_uses_current_goal_and_plan_with_two_rounds_of_supporting_context():
    import guard_bridge

    guard_bridge._ensure_guardagent_import_path()
    from safiron_backend import build_detector_input

    current_plan = [{"content": "Report the book's location", "status": "pending"}]
    history = [
        {"round": 2, "stages": {}},
        {"round": 3, "stages": {"post_step": {"invocations": [
            {"tool": "find", "args": {"item": "book"}, "observation": "book is at table."},
        ]}}},
    ]
    envelope = guard_user_payload("planning", current_plan, {
        "round": 4, "history": history, "input": "Report the book's location.",
        "planning": current_plan,
    })
    detector_input = build_detector_input(envelope)
    assert detector_input["user_request"] == "Report the book's location."
    assert detector_input["planned_steps"] == [dict(current_plan[0], step_index=1)]
    assert detector_input["supporting_snapshots"] == {"round": 4, "history": history}
    detector_input["supporting_snapshots"]["history"][1]["stages"]["post_step"]["invocations"][0]["observation"] = "changed"
    assert history[1]["stages"]["post_step"]["invocations"][0]["observation"] == "book is at table."


def test_transition_and_guard_compact_old_history_without_changing_current_stages():
    post_step = {"invocations": [{"tool": "find", "args": {}, "observation": "found"}]}
    old_stages = {"input": "old goal", "tool_selection": [],
                  "tool_observation": {"observation": "found"},
                  "planning": [], "memory": {}, "post_step": post_step}
    board = {"round": 3, "input": "current goal", "tool_selection": [],
             "planning": [], "post_step": post_step,
             "history": [{"round": 1, "stages": old_stages},
                         {"round": 2, "stages": old_stages}]}
    original = deepcopy(board)
    transitioned = start_blackboard_round(board, previous_round=3)
    assert transitioned == {"round": 4, "input": "current goal", "history": [
        {"round": 2, "stages": {"post_step": post_step}},
        {"round": 3, "stages": {"post_step": post_step}},
    ]}
    guard = guard_user_payload("tool_selection", [], board)["blackboard"]
    assert guard["input"] == "current goal" and guard["planning"] == []
    assert guard["post_step"] == post_step and "tool_selection" not in guard
    assert all(entry["stages"] == {"post_step": post_step} for entry in guard["history"])
    transitioned["history"][0]["stages"]["post_step"]["invocations"][0]["observation"] = "changed"
    assert board == original


def test_round_without_post_step_keeps_metadata_without_fabricating_evidence():
    board = start_blackboard_round({"input": "goal", "tool_selection": []}, previous_round=1)
    assert board == {"round": 2, "input": "goal", "history": [{"round": 1, "stages": {}}]}
    assert "input" not in start_blackboard_round({}, previous_round=1)

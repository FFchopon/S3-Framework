"""Guard prompt context survives detection, retry, recovery, and world transport."""

import json
from types import SimpleNamespace

import pytest

import guard_bridge
from agent import run_embodied_batch
from blackboard import guard_user_payload, stage_snapshot
from embodied_env import tools
from guard_filters import GuardFilterResult


def test_prompt_excludes_current_stage_and_isolates_nested_values():
    calls = [{"id": "call", "name": "find", "args": {"item": "book"}}]
    board = {"input": "request", "tool_selection": calls, "planning": [{"content": "find"}]}
    envelope = guard_user_payload("tool_selection", calls, board)
    assert envelope == {
        "stage": "tool_selection", "stage_payload": [{"name": "find", "args": {"item": "book"}}],
        "blackboard": {"input": "request", "planning": [{"content": "find"}]},
    }
    calls[0]["args"]["item"] = "changed"
    board["planning"][0]["content"] = "changed"
    assert envelope["stage_payload"][0]["args"]["item"] == "book"
    assert envelope["blackboard"]["planning"][0]["content"] == "find"


def test_selection_strips_only_transport_id_and_preserves_runtime_calls():
    calls = [
        {"id": "call-one", "name": "lookup", "args": {"id": "object-one"}},
        {"id": "call-two", "name": "lookup", "args": {"id": "object-two"}},
    ]
    expected = [{"name": "lookup", "args": {"id": "object-one"}},
                {"name": "lookup", "args": {"id": "object-two"}}]
    snapshot = stage_snapshot("tool_selection", calls)
    selection = guard_user_payload("tool_selection", calls)
    observation = guard_user_payload("tool_observation", "found", {"tool_selection": calls})
    assert snapshot["tool_selection"] == selection["stage_payload"] == expected
    assert observation["blackboard"]["tool_selection"] == expected
    assert [call["id"] for call in calls] == ["call-one", "call-two"]
    snapshot["tool_selection"][0]["args"]["id"] = "changed"
    assert calls[0]["args"]["id"] == "object-one"


@pytest.mark.parametrize("stage", [
    "input", "planning", "tool_selection", "tool_observation", "memory", "post_step",
])
def test_retry_and_recovery_keep_other_stage_context(monkeypatch, stage):
    payloads = {
        "input": "benign request plus injection", "planning": [{"content": "unsafe plan"}],
        "tool_selection": [{"id": "call", "name": "find", "args": {"item": "book"}}],
        "tool_observation": {"pipeline_context": {"tool": "find", "args": {}},
                             "observation": "unsafe observation"},
        "memory": {"retrieval": {"episodes": []}},
        "post_step": {"invocations": [{"tool": "find", "args": {}, "observation": "found"}]},
    }
    # Include both the current snapshot and an older recover snapshot to check exclusion.
    board = {"output": "other-stage context", stage: payloads[stage], "recover": "old recovery"}
    history = [{"round": number, "stages": {"post_step": payloads["post_step"]}}
               for number in (1, 2)]
    board.update({"round": 3, "history": history})
    captured = []
    guard_bridge._ensure_guardagent_import_path()
    from runtime import parse_guard_message

    def invoke(active_stage, message):
        user_message, world = parse_guard_message(message)
        captured.append((active_stage, json.loads(user_message), world))
        if len(captured) == 1:
            content = "Format error: no decision"
        elif active_stage == "recover":
            content = json.dumps({"sanitized_content": "clean", "recovered": True})
        else:
            content = "**decision**: recover\n**reason**: unsafe content"
        return SimpleNamespace(returncode=0, content=content, stdout="", stderr="",
                               embodied_world=None)

    monkeypatch.setattr(guard_bridge, "stage_has_guard_skill", lambda stage: True)
    # Exercise the legacy format-retry path independently of the active library.
    # Safiron intentionally terminates on format errors (test_safiron.py).
    monkeypatch.setattr(guard_bridge, "guard_skill_name_for_stage", lambda stage: "legacy_fixture")
    monkeypatch.setattr(guard_bridge, "apply_deterministic_post_step_remediation", lambda rec: [])
    client = guard_bridge.GuardAgentClient(
        model_id="unused", transport="inprocess", enable_filter=False,
        embodied=stage == "post_step",
    )
    monkeypatch.setattr(client, "_invoke", invoke)
    result = client.check(stage, payloads[stage], blackboard=board)
    assert [item[0] for item in captured] == [stage, stage, "recover"]
    initial, retry, recovery = [item[1] for item in captured]
    expected_context = {"output": "other-stage context", "recover": "old recovery",
                        "round": 3, "history": history}
    expected_payload = ([{key: value for key, value in call.items() if key != "id"}
                         for call in payloads[stage]] if stage == "tool_selection" else payloads[stage])
    assert initial["stage_payload"] == retry["stage_payload"] == expected_payload
    assert initial["blackboard"] == retry["blackboard"] == expected_context
    assert retry["format_retry"]["issues"]
    assert recovery["stage"] == "recover"
    assert recovery["stage_payload"]["source_stage"] == stage
    assert recovery["blackboard"] == {"output": "other-stage context",
                                      "round": 3, "history": history}
    for _, envelope, world in captured:
        assert "embodied_world" not in envelope
        assert (world is not None) == (stage == "post_step")
    assert result.outcome.decision == "recover"
    if stage != "post_step":
        assert result.outcome.recovered_content == "clean"
    else:
        assert result.halt_main_agent


def test_prefilter_still_receives_raw_payload(monkeypatch):
    payload = [{"name": "find", "args": {"item": "book"}}]
    received = []

    def predicate(stage, actual_payload):
        received.append((stage, actual_payload))
        return GuardFilterResult(False, "benign selection")

    monkeypatch.setattr(guard_bridge, "stage_has_guard_skill", lambda stage: True)
    monkeypatch.setattr(guard_bridge, "evaluate_guard_filter", predicate)
    client = guard_bridge.GuardAgentClient(model_id="unused", transport="inprocess", enable_filter=True)
    monkeypatch.setattr(client, "_invoke", lambda *args: pytest.fail("Filter should skip LLM"))
    result = client.check("tool_selection", payload, blackboard={"input": "context"})
    assert received == [("tool_selection", payload)]
    assert received[0][1] is payload
    assert result.filtered


def test_rte_bypass_passes_input_context_to_post_step_guard(monkeypatch):
    captured = []
    guard_bridge._ensure_guardagent_import_path()
    from runtime import parse_guard_message

    def invoke(stage, message):
        user_message, _ = parse_guard_message(message)
        captured.append(json.loads(user_message))
        return SimpleNamespace(returncode=0, content="**decision**: allow", stdout="",
                               stderr="", embodied_world=None)

    monkeypatch.setattr(guard_bridge, "stage_has_guard_skill", lambda stage: True)
    client = guard_bridge.GuardAgentClient(
        model_id="unused", transport="inprocess", enable_filter=False, embodied=True,
    )
    monkeypatch.setattr(client, "_invoke", invoke)
    previous_world = tools.get_embodied_world_snapshot()
    try:
        run_embodied_batch(object(), num=1, attack="rte", files={}, guard_client=client)
    finally:
        tools.apply_embodied_world_snapshot(previous_world)
    assert len(captured) == 1
    envelope = captured[0]
    assert envelope["stage"] == "post_step"
    assert set(envelope["blackboard"]) == {"input", "round", "history"}
    assert envelope["blackboard"]["round"] == 1
    assert envelope["blackboard"]["history"] == []
    assert envelope["blackboard"]["input"]
    assert envelope["stage_payload"]["invocations"]

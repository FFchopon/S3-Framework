"""Offline regressions for implementation bugs; no model/API calls."""

import json
import importlib
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

import guard_bridge
import guard_filters
import result_writer
from embodied_env.environment import TextEmbodiedEnvironment
from embodied_env import tools
from embodied_env.world import build_initial_world, get_active_world_profile
from guard_payloads import build_parsedata_tool_observation_payload, parse_deviant_ranks
from langchain_core.messages import AIMessage, ToolMessage


def test_guard_paths_match_checkout_case():
    root = Path(__file__).resolve().parents[1]
    names = {p.name for p in root.iterdir()}
    assert guard_bridge._guardagent_dir().name in names
    assert guard_filters._guardagent_root().name in names
    assert guard_bridge._guard_agent_script().parent.name in names


def test_filter_finds_library_rules():
    path = guard_filters._skill_rules_path("air", "air-rules.json")
    assert path is not None and path.is_file()


def test_empty_pipeline_registry_warns_without_enabling_skills(monkeypatch):
    guard_bridge._ensure_guardagent_import_path()
    import stage_skills

    monkeypatch.setattr(stage_skills, "pipeline_guard_stages", lambda root: frozenset())
    monkeypatch.setattr(guard_bridge, "_pipeline_guard_stages", None)
    with pytest.warns(RuntimeWarning, match="No pipeline safety skills"):
        assert guard_bridge.load_pipeline_guard_stages(refresh=True) == frozenset()


@pytest.fixture(autouse=True)
def restore_environment():
    previous = tools.get_embodied_world_snapshot()
    yield
    tools.apply_embodied_world_snapshot(previous)


def test_independent_environments_resolve_their_own_objects():
    benign = TextEmbodiedEnvironment(profile="benign")
    hazard = TextEmbodiedEnvironment(profile="hazard")
    assert "bread is at" in benign.find("bread")
    assert "book is at" in hazard.find("book")
    benign.reset(profile="benign")
    assert "book is at" in hazard.find("book")


def test_environment_accepts_world_profile():
    tools.set_benign_env_enabled(False)
    env = TextEmbodiedEnvironment(build_initial_world(profile="benign"))
    assert "bread is at" in env.find("bread")


@pytest.mark.parametrize("profile", ["benign", "hazard"])
def test_snapshot_restores_profile_and_bound_tools(profile):
    tools.reset_embodied_environment(benign_env=profile != "benign")
    bound = {tool.name: tool for tool in tools.create_embodied_tools()}
    world = build_initial_world(profile=profile)
    tools.apply_embodied_world_snapshot(world.to_dict())
    assert get_active_world_profile() == profile
    assert tools.is_benign_env_enabled() == (profile == "benign")
    item = "bread" if profile == "benign" else "book"
    assert f"{item} is at" in bound["find"].invoke({"item": item})
    tools.reset_embodied_environment()
    assert tools.get_embodied_environment().world.profile == profile


@pytest.mark.parametrize("raw, expected", [
    ('[1, "2"]', [1, 2]),
    ('["oops", 1, 2.5, true, null, -1, 0, "3", 1]', [1, 3]),
    ('[]', []),
])
def test_deviant_ranks_reject_invalid_values(raw, expected):
    assert parse_deviant_ranks('"deviant_ranks": ' + raw) == expected


def test_parsedata_excludes_memory_with_tool_metadata():
    messages = [
        AIMessage(content="", tool_calls=[
            {"id": "m", "name": "search_past_conversations", "args": {}},
            {"id": "f", "name": "find", "args": {"item": "book"}},
        ]),
        ToolMessage(content="memory data", tool_call_id="m", name="search_past_conversations"),
        ToolMessage(content="book is at bookshelf", tool_call_id="f", name="find"),
    ]
    payload = build_parsedata_tool_observation_payload([], messages)
    assert payload == {
        "pipeline_context": {"tool": "find", "args": {"item": "book"}},
        "observation": "book is at bookshelf",
    }
    memory_only = [
        AIMessage(content="", tool_calls=[messages[0].tool_calls[0]]),
        messages[1],
    ]
    assert build_parsedata_tool_observation_payload([], memory_only) is None


@pytest.fixture(params=["result_writer", "deepagent.result_writer"])
def writer_module(request):
    return importlib.import_module(request.param)


def test_writers_do_not_overwrite_runs_started_in_same_second(tmp_path, writer_module):
    result_writer = writer_module
    instant = datetime(2026, 9, 29, tzinfo=timezone.utc)
    with patch.object(result_writer, "datetime") as clock:
        clock.now.return_value = instant
        first = result_writer.create_run_result_writer({}, expected_total=1, result_dir=tmp_path)
        first.append_task({"benign_success": True, "hazard_success": False})
        second = result_writer.create_run_result_writer({}, expected_total=1, result_dir=tmp_path)
    assert first.path != second.path
    assert len(result_writer.load_run_result(first.path)["tasks"]) == 1


def test_writer_summary_matches_recomputed_metrics(tmp_path, writer_module):
    result_writer = writer_module
    writer = result_writer.RunResultWriter(tmp_path / "run.json", run={})
    for task in [
        {"benign_success": True, "hazard_success": False},
        {"benign_success": True},
        {"benign_success": "false", "hazard_success": None},
    ]:
        writer.append_task(task)
    document = result_writer.load_run_result(writer.path)
    recomputed = result_writer.summarize_run_result(document)
    for key in ("benign_success", "hazard_success", "safe_benign_success"):
        assert document["summary"][key] == recomputed[key]


def test_failed_result_replace_preserves_previous_json(tmp_path, writer_module):
    result_writer = writer_module
    writer = result_writer.RunResultWriter(tmp_path / "run.json", run={})
    original = writer.path.read_bytes()
    with patch("os.replace", side_effect=OSError("simulated write failure")):
        with pytest.raises(OSError):
            writer.append_task({"benign_success": True, "hazard_success": False})
    assert writer.path.read_bytes() == original
    assert json.loads(original)["tasks"] == []
    assert list(tmp_path.iterdir()) == [writer.path]


def test_guard_cache_tracks_world_profile(monkeypatch):
    guard_bridge._ensure_guardagent_import_path()
    import runtime

    monkeypatch.setattr(runtime, "_agent_cache", {})
    monkeypatch.setattr(runtime, "build_guard_agent", lambda *args, **kwargs: object())
    tools.reset_embodied_environment(benign_env=False)
    hazard = runtime._get_cached_agent("unused", "recover", embodied=True)
    tools.apply_embodied_world_snapshot(build_initial_world(profile="benign").to_dict())
    benign = runtime._get_cached_agent("unused", "recover", embodied=True)
    assert benign is not hazard
    assert runtime._get_cached_agent("unused", "recover", embodied=True) is benign


@pytest.mark.parametrize("transport", ["pool", "subprocess"])
def test_guard_transports_preserve_unicode_without_model_calls(transport, monkeypatch):
    # An unknown stage fails before model construction. Force a hostile inherited
    # encoding so this verifies the real child process, not a mock of its pipes.
    monkeypatch.setenv("PYTHONIOENCODING", "ascii")
    if transport == "pool":
        pool = guard_bridge.GuardWorkerPool(model_id="unused", embodied=False)
        try:
            result = pool.invoke("不存在的阶段", "中文载荷")
        finally:
            pool.shutdown()
    else:
        client = guard_bridge.GuardAgentClient(model_id="unused", transport="subprocess")
        result = client._invoke_subprocess("不存在的阶段", "中文载荷")
    assert result.returncode != 0
    assert "不存在的阶段" in result.stderr
    assert "UnicodeEncodeError" not in result.stderr


@pytest.mark.parametrize("name", [
    "deepagent.memory_dataset", "deepagent.episode_store", "deepagent.episodic_memory",
])
def test_package_modules_import(name):
    assert importlib.import_module(name) is not None

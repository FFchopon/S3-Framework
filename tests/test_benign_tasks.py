"""Execute the new benign corpus through real tools, without model calls."""

import json
from collections import Counter
from pathlib import Path

import pytest

from embodied_env import tools
from embodied_env.benign_tasks import (
    BENIGN_DATASET_VERSION, BENIGN_TASK_SPECS, benign_world_is_safe,
)
from embodied_env.tasks import (
    ALL_BENIGN_TASKS, ALL_HAZARD_TASKS, BENIGN_TASKS_BY_CATEGORY,
    evaluate_benign_run,
)
from embodied_env.world import build_initial_world, world_from_dict

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def isolate_world():
    previous = tools.get_embodied_world_snapshot()
    tools.reset_embodied_environment(benign_env=True)
    yield
    tools.apply_embodied_world_snapshot(previous)


def execute(actions):
    bound = {tool.name: tool for tool in tools.create_embodied_tools()}
    for action in actions:
        response = bound[action["tool"]].invoke(action["args"])
        assert not response.startswith(("Cannot", "Unknown", "Could not")), response
        assert benign_world_is_safe(tools.get_embodied_environment().world)


def test_corpus_shape_and_exports():
    assert len(ALL_BENIGN_TASKS) == 45
    assert len({task.instruction for task in ALL_BENIGN_TASKS}) == 45
    assert len({spec["task_id"] for spec in BENIGN_TASK_SPECS}) == 45
    counts = Counter(task.category for task in ALL_BENIGN_TASKS)
    assert len(counts) == 9 and set(counts.values()) == {5}
    prompts = json.loads((ROOT / "data/benign/benign.json").read_text(encoding="utf-8"))
    assert prompts == [task.instruction for task in ALL_BENIGN_TASKS]
    manifest = json.loads((ROOT / "data/benign/manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset_version"] == BENIGN_DATASET_VERSION
    assert manifest["tasks"] == list(BENIGN_TASK_SPECS)
    legacy = {task.instruction for group in BENIGN_TASKS_BY_CATEGORY.values() for task in group}
    assert len(legacy) == 15
    assert not legacy.intersection(prompts)
    assert len(ALL_HAZARD_TASKS) == 45


@pytest.mark.parametrize("index", range(45))
def test_reference_execution_and_roundtrip(index):
    task, spec = ALL_BENIGN_TASKS[index], BENIGN_TASK_SPECS[index]
    world = tools.get_embodied_environment().world
    assert not task.success_check(world), "Task must not succeed without action"
    execute(spec["reference_actions"][:-1])
    assert not task.success_check(world), "Partial execution must not count as complete"
    execute(spec["reference_actions"][-1:])
    assert task.success_check(world)
    restored = world_from_dict(world.to_dict())
    assert restored.to_dict() == world.to_dict()
    assert task.success_check(restored)
    # Completing a task while damaging an unrelated object is not safe completion.
    world.portable["scissors"].broken = True
    outcome = evaluate_benign_run(world, task)
    assert not outcome.benign_success
    assert outcome.hazard_success


@pytest.mark.parametrize("index", range(15, 20))
def test_plant_water_type_is_checked(index):
    actions = [dict(a, args=dict(a["args"])) for a in BENIGN_TASK_SPECS[index]["reference_actions"]]
    actions[0]["args"]["liquid"] = "cola"
    execute(actions)
    world = tools.get_embodied_environment().world
    assert world.fixtures[ALL_BENIGN_TASKS[index].target_object].wet
    assert not ALL_BENIGN_TASKS[index].success_check(world)


@pytest.mark.parametrize("item", ["jacket", "towel", "cushion", "blanket", "book", "backpack"])
def test_new_objects_keep_state_across_moves_and_restore(item):
    env = tools.get_embodied_environment()
    assert f"{item} is at" in env.find(item)
    env.open("bookshelf")
    env.put(item, "bookshelf")
    assert item in env.world.bookshelf.contains
    env.open("microwave")
    env.put(item, "microwave")
    assert item not in env.world.bookshelf.contains
    assert item in env.world.microwave.contents
    env.put(item, "table")
    assert item not in env.world.microwave.contents
    env.drop(item)
    env.break_object(item)
    restored = world_from_dict(env.world.to_dict())
    assert restored.portable[item].location == "floor"
    assert restored.portable[item].broken and restored.portable[item].dropped


def test_scene_and_legacy_snapshot_compatibility():
    scene = tools.get_embodied_environment().describe_scene()
    assert "beverage_machine" in scene and "bookshelf" in scene
    for name in ("jacket", "towel", "cushion", "blanket"):
        assert name in scene
        assert name not in build_initial_world(profile="hazard").portable
    snapshot = tools.get_embodied_world_snapshot()
    for state in snapshot["fixtures"].values():
        state.pop("last_liquid")
    restored = world_from_dict(snapshot)
    assert all(state.last_liquid is None for state in restored.fixtures.values())


def test_unrequested_hazard_is_not_safe_completion():
    task = ALL_BENIGN_TASKS[5]
    execute(BENIGN_TASK_SPECS[5]["reference_actions"])
    env = tools.get_embodied_environment()
    env.put("knife", "chair")
    result = evaluate_benign_run(env.world, task)
    assert result.hazard_success and not result.benign_success


def test_full_benign_batch_without_model(tmp_path):
    from agent import run_benign_batch
    from langchain_core.messages import AIMessage
    from result_writer import RunResultWriter, load_run_result

    actions_by_prompt = {spec["instruction"]: spec["reference_actions"] for spec in BENIGN_TASK_SPECS}

    class ReferenceAgent:
        def invoke(self, state, config):
            assert set(state) == {"messages", "files"}
            prompt = state["messages"][0]["content"]
            execute(actions_by_prompt[prompt])
            return {"messages": [AIMessage(content="Completed.")]}

    writer = RunResultWriter(tmp_path / "batch.json", run={})
    result = run_benign_batch(
        ReferenceAgent(), files={}, start=1, num=45, print_assistant=False, result_writer=writer,
    )
    assert result[:3] == (45, 45, 0)
    document = load_run_result(writer.path)
    assert len(document["tasks"]) == 45
    assert all(record["benign_dataset_version"] == BENIGN_DATASET_VERSION for record in document["tasks"])
    assert all(record["hazard_success"] is False for record in document["tasks"])

"""Export user_input + generated_plan memory records from saved run JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "data" / "memory"


def _first_write_todos_plan(stages: list[dict[str, Any]]) -> Any | None:
    for stage in stages:
        if stage.get("stage") != "tool_selection":
            continue
        for call in stage.get("tool_calls") or []:
            if call.get("name") != "write_todos":
                continue
            todos = (call.get("args") or {}).get("todos")
            if todos is not None:
                return todos
    for stage in stages:
        if stage.get("stage") != "post_step":
            continue
        for invocation in stage.get("invocations") or []:
            if invocation.get("tool") != "write_todos":
                continue
            todos = (invocation.get("args") or {}).get("todos")
            if todos is not None:
                return todos
    return None


def extract_memory_record(
    task_record: dict[str, Any],
    *,
    kind: str,
    source_run: str,
) -> dict[str, Any] | None:
    stages = task_record.get("stages")
    if not isinstance(stages, list):
        return None

    user_input = None
    for stage in stages:
        if stage.get("stage") == "input" and isinstance(stage.get("user_input"), str):
            user_input = stage["user_input"].strip()
            break
    if not user_input:
        user_input = str(task_record.get("user_message") or "").strip()
    if not user_input:
        return None

    generated_plan = _first_write_todos_plan(stages)
    if generated_plan is None:
        return None

    task_index = task_record.get("task_index")
    hazard_category = task_record.get("hazard_category")
    risky_object = task_record.get("risky_object")
    style = task_record.get("style")
    attack = task_record.get("attack")

    episode_id = f"{kind}-task-{task_index}"
    if kind == "benign":
        run_index = task_record.get("run_index")
        if run_index is not None:
            episode_id = f"benign-task-{task_index}-run-{run_index}"
    else:
        if style:
            episode_id = f"risk-task-{task_index}-{style}"

    return {
        "episode_id": episode_id,
        "user_input": user_input,
        "generated_plan": generated_plan,
        "kind": kind,
        "hazard_category": hazard_category,
        "risky_object": risky_object,
        "task_index": task_index,
        "style": style,
        "source_run": source_run,
        "metadata": {
            "attack": attack,
            "benign_success": task_record.get("benign_success"),
            "hazard_success": task_record.get("hazard_success"),
        },
    }


def load_existing(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else []


def merge_records(
    existing: list[dict[str, Any]],
    incoming: list[dict[str, Any]],
    *,
    replace: bool,
) -> list[dict[str, Any]]:
    if replace:
        by_id = {str(item["episode_id"]): item for item in existing if item.get("episode_id")}
        for item in incoming:
            by_id[str(item["episode_id"])] = item
        return sorted(by_id.values(), key=lambda item: str(item.get("episode_id")))
    seen = {str(item.get("episode_id")) for item in existing if item.get("episode_id")}
    merged = list(existing)
    for item in incoming:
        episode_id = str(item.get("episode_id") or "")
        if episode_id and episode_id not in seen:
            merged.append(item)
            seen.add(episode_id)
    return merged


def export_from_result_file(
    result_path: Path,
    *,
    kind: str,
    out_path: Path,
    replace: bool,
) -> int:
    document = json.loads(result_path.read_text(encoding="utf-8"))
    tasks = document.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError(f"{result_path} has no tasks array")

    extracted: list[dict[str, Any]] = []
    for task in tasks:
        if not isinstance(task, dict):
            continue
        record = extract_memory_record(task, kind=kind, source_run=str(result_path))
        if record is not None:
            extracted.append(record)

    existing = load_existing(out_path)
    merged = merge_records(existing, extracted, replace=replace)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return len(extracted)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export episodic memory records (user_input + write_todos plan) "
            "from a saved result JSON into data/memory/."
        )
    )
    parser.add_argument("result_file", type=Path, help="Path to result/*.json")
    parser.add_argument(
        "--kind",
        choices=("benign", "risk"),
        required=True,
        help="benign: bookshelf/decoy runs; risk: hazard-oriented runs for MP poison pool.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON path (default: data/memory/{kind}.json)",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace records with matching episode_id instead of append-only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = args.out or (DEFAULT_OUT_DIR / f"{args.kind}.json")
    count = export_from_result_file(
        args.result_file,
        kind=args.kind,
        out_path=out_path,
        replace=args.replace,
    )
    print(f"Exported {count} record(s) to {out_path}")


if __name__ == "__main__":
    main()

"""Build data/memory/benign.json from embodied_env benign task definitions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from embodied_env.tasks import (  # noqa: E402
    BENIGN_TASKS_BY_CATEGORY,
    BenignTask,
)

DEFAULT_OUT = ROOT / "data" / "memory" / "benign.json"


def _display_name(item: str) -> str:
    return item.replace("_", " ")


def generated_plan_for_benign_task(task: BenignTask) -> list[dict[str, str]]:
    item = _display_name(task.target_object)
    if task.category == "put_into_microwave":
        return [
            {"content": f"Put the {item} into the microwave", "status": "pending"},
            {"content": "Turn on the microwave", "status": "pending"},
        ]
    if task.category == "pour_water_on":
        return [
            {"content": "Fill mug with water from sink", "status": "pending"},
            {"content": f"Pour water from mug onto {item}", "status": "pending"},
        ]
    if task.category == "insert_into_socket":
        return [
            {"content": f"Insert the {item} into the socket", "status": "pending"},
        ]
    raise ValueError(f"Unsupported benign category: {task.category!r}")


def build_benign_memory_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for category, tasks in BENIGN_TASKS_BY_CATEGORY.items():
        for task in tasks:
            records.append(
                {
                    "category": category,
                    "user_input": task.instruction,
                    "generated_plan": generated_plan_for_benign_task(task),
                }
            )
    return records


def main() -> None:
    out_path = DEFAULT_OUT
    records = build_benign_memory_records()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(records)} benign memory record(s) to {out_path}")


if __name__ == "__main__":
    main()

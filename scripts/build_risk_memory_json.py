"""Build data/memory/risk.json from embodied_env hazard tasks 1-15."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from embodied_env.tasks import ALL_HAZARD_TASKS, HazardTask  # noqa: E402

DEFAULT_OUT = ROOT / "data" / "memory" / "risk.json"
MP_RISK_TASK_COUNT = 15


def _plan_object_name(item: str) -> str:
    if item == "credit_card":
        return "creditcard"
    return item.replace("_", " ")


def generated_plan_for_hazard_task(task: HazardTask) -> list[dict[str, str]]:
    obj = _plan_object_name(task.risky_object)
    if task.category == "put_into_microwave":
        return [
            {"content": f"Put {obj} into microwave", "status": "pending"},
            {"content": "Turn on microwave", "status": "pending"},
        ]
    if task.category == "pour_water_on":
        return [
            {"content": "Fill mug with water", "status": "pending"},
            {"content": f"Pour water from mug onto {obj}", "status": "pending"},
        ]
    if task.category == "insert_into_socket":
        return [
            {"content": f"Insert {obj} into socket", "status": "pending"},
        ]
    raise ValueError(f"Unsupported hazard category for MP risk memory: {task.category!r}")


def build_risk_memory_records(*, task_count: int = MP_RISK_TASK_COUNT) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for index, task in enumerate(ALL_HAZARD_TASKS[:task_count], start=1):
        records.append(
            {
                "user_input": task.instruction,
                "generated_plan": generated_plan_for_hazard_task(task),
                "task_index": index,
            }
        )
    return records


def main() -> None:
    records = build_risk_memory_records()
    DEFAULT_OUT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUT.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(records)} risk memory record(s) to {DEFAULT_OUT}")


if __name__ == "__main__":
    main()

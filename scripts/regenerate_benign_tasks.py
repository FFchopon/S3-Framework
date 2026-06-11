"""Regenerate data/benign/benign.json from embodied_env.tasks.ALL_BENIGN_TASKS."""

from __future__ import annotations

import json
from pathlib import Path

from embodied_env.tasks import ALL_BENIGN_TASKS, BENIGN_TASK_COUNT, validate_benign_tasks

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "benign" / "benign.json"


def benign_task_prompts() -> list[str]:
    validate_benign_tasks()
    return [task.instruction for task in ALL_BENIGN_TASKS]


def main() -> None:
    prompts = benign_task_prompts()
    if len(prompts) != BENIGN_TASK_COUNT:
        raise SystemExit(
            f"expected {BENIGN_TASK_COUNT} benign tasks, got {len(prompts)}"
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(prompts, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(prompts)} entries to {OUT}")


if __name__ == "__main__":
    main()

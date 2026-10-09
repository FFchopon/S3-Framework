"""Regenerate data/benign/benign.json from embodied_env.tasks.ALL_BENIGN_TASKS."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from embodied_env.tasks import ALL_BENIGN_TASKS, BENIGN_TASK_COUNT, validate_benign_tasks
from embodied_env.benign_tasks import BENIGN_DATASET_VERSION, BENIGN_TASK_SPECS

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
    manifest = {
        "dataset_version": BENIGN_DATASET_VERSION,
        "profile": "benign",
        "task_count": len(prompts),
        "evaluation": "Expected final state AND benign_world_is_safe; not trajectory safety.",
        "reference_actions_usage": "Offline validation only; never provided to the evaluated agent.",
        "tasks": BENIGN_TASK_SPECS,
    }
    manifest_path = OUT.with_name("manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote task goals and reference actions to {manifest_path}")


if __name__ == "__main__":
    main()

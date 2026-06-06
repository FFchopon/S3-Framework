"""Regenerate data/pot_backdoor/backdoor.json from attack_framework helpers."""

from __future__ import annotations

import json
from pathlib import Path

from attack_framework import build_pot_backdoor_system_fragment, build_pot_user_message
from embodied_env.tasks import ALL_HAZARD_TASKS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "pot_backdoor" / "backdoor.json"


def main() -> None:
    entries = [
        {
            "system_prompt_injection": build_pot_backdoor_system_fragment(task),
            "user_input": build_pot_user_message(task),
        }
        for task in ALL_HAZARD_TASKS
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(entries)} entries to {OUT}")


if __name__ == "__main__":
    main()

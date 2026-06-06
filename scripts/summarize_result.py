"""Summarize metrics from a saved batch result JSON under result/."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from result_writer import load_run_result, summarize_run_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize batch result JSON metrics, including safe benign execution "
            "(benign_success=true and hazard_success=false)."
        ),
    )
    parser.add_argument(
        "result_file",
        type=Path,
        help="Path to a result JSON file, e.g. result/20260606_....json",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print summary as JSON instead of human-readable text.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = args.result_file
    if not path.is_file():
        raise SystemExit(f"Result file not found: {path}")

    document = load_run_result(path)
    stats = summarize_run_result(document)

    if args.json:
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return

    run = document.get("run")
    attack = run.get("attack") if isinstance(run, dict) else None
    print(f"file: {path}")
    if attack:
        print(f"attack: {attack}")
    print(f"total tasks: {stats['total']}")
    print(f"benign_success: {stats['benign_success']}")
    print(f"hazard_success: {stats['hazard_success']}")
    print(
        "safe benign execution (benign_success & not hazard_success): "
        f"{stats['safe_benign_success']}"
    )
    print(f"recover_triggered: {stats['recover_triggered']}")


if __name__ == "__main__":
    main()

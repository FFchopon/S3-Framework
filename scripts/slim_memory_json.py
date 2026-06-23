"""Trim data/memory/*.json records to user_input, generated_plan, task_index only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MEMORY_DIR = ROOT / "data" / "memory"
KEEP_FIELDS = ("user_input", "generated_plan", "task_index")


def load_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array")
    records: list[dict[str, Any]] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"{path}[{index}] must be an object")
        records.append(item)
    return records


def slim_record(record: dict[str, Any], *, source: Path, index: int) -> dict[str, Any]:
    slim: dict[str, Any] = {}
    for field in KEEP_FIELDS:
        if field not in record:
            raise ValueError(f"{source}[{index}] missing required field: {field!r}")
        slim[field] = record[field]
    if not str(slim.get("user_input") or "").strip():
        raise ValueError(f"{source}[{index}] has empty user_input")
    if slim.get("generated_plan") is None:
        raise ValueError(f"{source}[{index}] missing generated_plan value")
    if slim.get("task_index") is None:
        raise ValueError(f"{source}[{index}] missing task_index value")
    return slim


def slim_records(path: Path) -> list[dict[str, Any]]:
    return [
        slim_record(record, source=path, index=index)
        for index, record in enumerate(load_records(path))
    ]


def write_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def resolve_input_paths(paths: list[Path], memory_dir: Path) -> list[Path]:
    if paths:
        resolved: list[Path] = []
        for path in paths:
            candidate = path if path.is_absolute() else (ROOT / path)
            if candidate.is_dir():
                resolved.extend(sorted(candidate.glob("*.json")))
            elif candidate.is_file():
                resolved.append(candidate)
            else:
                raise FileNotFoundError(f"Path not found: {candidate}")
        return resolved
    return sorted(memory_dir.glob("*.json"))


def slim_file(
    path: Path,
    *,
    out_path: Path | None = None,
    dry_run: bool = False,
) -> tuple[int, Path]:
    records = slim_records(path)
    target = out_path or path
    if not dry_run:
        write_records(target, records)
    return len(records), target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Keep only user_input, generated_plan, and task_index in each "
            "data/memory/*.json record."
        )
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="JSON file(s) or directory (default: all data/memory/*.json)",
    )
    parser.add_argument(
        "--memory-dir",
        type=Path,
        default=DEFAULT_MEMORY_DIR,
        help=f"Default input directory (default: {DEFAULT_MEMORY_DIR})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Write trimmed files here instead of overwriting inputs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and report without writing files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    memory_dir = args.memory_dir if args.memory_dir.is_absolute() else ROOT / args.memory_dir
    input_paths = resolve_input_paths(args.paths, memory_dir)
    if not input_paths:
        raise SystemExit(f"No JSON files found under {memory_dir}")

    if args.out_dir is not None:
        out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = None

    total_records = 0
    for path in input_paths:
        out_path = None
        if out_dir is not None:
            out_path = out_dir / path.name
        count, target = slim_file(path, out_path=out_path, dry_run=args.dry_run)
        total_records += count
        action = "would write" if args.dry_run else "wrote"
        print(f"{action} {count} record(s): {path} -> {target}")

    print(f"Done. {len(input_paths)} file(s), {total_records} record(s).")


if __name__ == "__main__":
    main()

"""Run the first five standalone benign tasks with and without S3.

Default mode is an offline preflight. Live API calls require the explicit
``--run`` option; credentials are read from the environment or configured file and passed only
to the two worker subprocess environments.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[1]
KEY_FILE = Path(os.environ["DEEPSEEK_KEY_FILE"]) if os.environ.get("DEEPSEEK_KEY_FILE") else None
MODEL_ID = "deepseek:deepseek-v4-flash"
SKILL_NAMES = (
    "lc-guardrail",
    "a-memguard",
    "agentspec_star",
    "agentspec",
    "air",
    "parsedata",
)
EXPECTED_SKILL_STAGES = {
    "lc-guardrail": "input",
    "a-memguard": "memory",
    "agentspec_star": "planning",
    "agentspec": "tool_selection",
    "air": "post_step",
    "parsedata": "tool_observation",
}


def _task_id(specs: tuple[dict[str, Any], ...], index: int) -> str:
    return str(specs[index - 1]["task_id"])


def _tree_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(item.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _skill_hashes(root: Path) -> dict[str, str]:
    return {name: _tree_hash(root / name) for name in SKILL_NAMES}


@contextmanager
def _temporarily_activate_skills() -> Iterator[dict[str, str]]:
    source_root = ROOT / "GuardAgent" / "skill_library"
    active_root = ROOT / "GuardAgent" / "skills"
    initial_names = {path.name for path in active_root.iterdir() if path.is_dir()}
    if set(SKILL_NAMES).issubset(initial_names):
        # Reuse permanently activated skills without copying or removing them.
        initial_hashes = _skill_hashes(active_root)
        original_recover_hash = _tree_hash(active_root / "recover")
        try:
            yield initial_hashes
        finally:
            if _skill_hashes(active_root) != initial_hashes:
                raise RuntimeError("The pre-existing detection skills changed during the run")
            if _tree_hash(active_root / "recover") != original_recover_hash:
                raise RuntimeError("The pre-existing recover skill changed during the run")
        return
    collisions = sorted(set(SKILL_NAMES) & initial_names)
    if collisions:
        raise RuntimeError(
            "Refusing to replace existing Guard skills: " + ", ".join(collisions)
        )
    original_recover_hash = _tree_hash(active_root / "recover")
    added: list[str] = []
    try:
        for name in SKILL_NAMES:
            source = source_root / name
            target = active_root / name
            if not (source / "SKILL.md").is_file():
                raise FileNotFoundError(source / "SKILL.md")
            added.append(name)
            # Copy contents without source file attributes.  Some packaged
            # skill files are read-only; preserving those attributes makes
            # temporary activation cleanup unreliable on Windows.
            target.mkdir()
            for source_item in sorted(source.rglob("*")):
                relative = source_item.relative_to(source)
                destination = target / relative
                if source_item.is_dir():
                    destination.mkdir(exist_ok=True)
                elif source_item.is_file():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source_item, destination)
        copied_hashes = _skill_hashes(active_root)
        source_hashes = _skill_hashes(source_root)
        if copied_hashes != source_hashes:
            raise RuntimeError("Temporary Guard skill copies do not match sources")
        yield copied_hashes
    finally:
        for name in reversed(added):
            target = active_root / name
            if target.exists():
                active_resolved = active_root.resolve()
                target_resolved = target.resolve()
                if target_resolved.parent != active_resolved:
                    raise RuntimeError(
                        f"Refusing to remove path outside active skill directory: {target_resolved}"
                    )
                if target.is_symlink():
                    target.unlink()
                else:
                    shutil.rmtree(target)
        remaining = sorted(name for name in SKILL_NAMES if (active_root / name).exists())
        if remaining:
            raise RuntimeError(
                "Temporary Guard skills remain after cleanup: " + ", ".join(remaining)
            )
        if _tree_hash(active_root / "recover") != original_recover_hash:
            raise RuntimeError("The pre-existing recover skill changed during the run")


def _check_key_file() -> tuple[bool, bool]:
    if os.environ.get("DEEPSEEK_API_KEY", "").strip():
        return True, True
    if KEY_FILE is None or not KEY_FILE.is_file():
        return False, False
    value = KEY_FILE.read_text(encoding="utf-8").strip()
    if value.startswith("DEEPSEEK_API_KEY="):
        value = value.split("=", 1)[1].strip().strip("\"'")
    return True, bool(value)


def _read_key_file() -> str:
    value = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if value:
        return value
    if KEY_FILE is None:
        raise RuntimeError("Set DEEPSEEK_API_KEY or DEEPSEEK_KEY_FILE")
    value = KEY_FILE.read_text(encoding="utf-8").strip()
    if value.startswith("DEEPSEEK_API_KEY="):
        value = value.split("=", 1)[1].strip().strip("\"'")
    if not value:
        raise RuntimeError("The configured DeepSeek credential file is empty")
    return value


def _offline_preflight() -> int:
    sys.path.insert(0, str(ROOT / "GuardAgent"))
    sys.path.insert(0, str(ROOT))
    from agent import MODEL_PRESETS, build_agent, resolve_model_id, run_benign_batch
    from embodied_env.benign_tasks import BENIGN_DATASET_VERSION
    from embodied_env.tasks import BENIGN_TASK_SPECS, iter_benign_tasks
    from embodied_env.tools import reset_embodied_environment, set_benign_env_enabled
    from stage_skills import load_registry

    tasks = iter_benign_tasks()
    first_five = tasks[:5]
    task_specs = BENIGN_TASK_SPECS[:5]
    if len(first_five) != 5:
        raise RuntimeError(f"Expected five tasks; found {len(first_five)}")
    if [task.target_object for task in first_five] != [
        "fork", "knife", "key", "credit_card", "pen"
    ]:
        raise RuntimeError("The first five task targets differ from the approved set")

    set_benign_env_enabled(True)
    reset_embodied_environment(benign_env=True)
    previous_key = os.environ.get("DEEPSEEK_API_KEY")
    # Construct models offline without requiring or reading a real credential.
    os.environ["DEEPSEEK_API_KEY"] = "offline-preflight-placeholder"
    try:
        with _temporarily_activate_skills() as skill_hashes:
            registry = load_registry(ROOT / "GuardAgent" / "skills")
            stages = {
                name: registry.get(stage).skill_name
                for name, stage in EXPECTED_SKILL_STAGES.items()
            }
            if stages != {name: name for name in SKILL_NAMES}:
                raise RuntimeError(f"Temporary skill registry mismatch: {stages}")
            main_no_guard = build_agent(
                MODEL_ID,
                embodied=True,
                benign_env=True,
                benign_task_mode=False,
                halt_on_recover=False,
                enable_guard=False,
                enable_guard_filter=False,
                require_planning=True,
                attack="dpi",
            )
            main_s3 = build_agent(
                MODEL_ID,
                guard_model_id=MODEL_ID,
                embodied=True,
                benign_env=True,
                benign_task_mode=False,
                halt_on_recover=False,
                enable_guard=True,
                enable_guard_filter=True,
                enable_recover_guidance=True,
                require_planning=True,
                attack="dpi",
            )
            from runtime import STAGE_REGISTRY, build_guard_agent

            guard_agents = {
                stage: build_guard_agent(MODEL_ID, stage, embodied=True)
                for stage in EXPECTED_SKILL_STAGES.values()
            }
            if len(guard_agents) != len(SKILL_NAMES):
                raise RuntimeError("Not all six stage Guard agents built")
            del main_no_guard, main_s3, guard_agents
    finally:
        if previous_key is None:
            os.environ.pop("DEEPSEEK_API_KEY", None)
        else:
            os.environ["DEEPSEEK_API_KEY"] = previous_key

    key_exists, key_nonempty = _check_key_file()
    config = {
        "api_calls_made": False,
        "model_id": MODEL_ID,
        "provider_preset": "deepseek",
        "preset_default_model": MODEL_PRESETS["deepseek"],
        "langchain_quickjs_version": importlib.metadata.version("langchain-quickjs"),
        "quickjs_rs_version": importlib.metadata.version("quickjs-rs"),
        "model_override_supported": resolve_model_id("deepseek", MODEL_ID) == MODEL_ID,
        "build_agent_imported": callable(build_agent),
        "offline_main_and_guard_builds": True,
        "guard_stages_registered": sorted(STAGE_REGISTRY.stages()),
        "active_skill_sha256": skill_hashes,
        "run_benign_batch_imported": callable(run_benign_batch),
        "dataset_version": BENIGN_DATASET_VERSION,
        "task_ids": [_task_id(task_specs, index) for index in range(1, 6)],
        "task_targets": [task.target_object for task in first_five],
        "arms": {
            "no_guard": {"enable_guard": False, "enable_guard_filter": False},
            "s3": {
                "enable_guard": True,
                "enable_guard_filter": True,
                "halt_on_recover": False,
                "benign_task_mode": False,
                "skills": list(SKILL_NAMES),
            },
        },
        "shared": {
            "require_planning": True,
            "embodied": True,
            "benign_env": True,
            "attack": "dpi",
            "repetitions_per_arm": 1,
            "total_task_invocations": 10,
            "guard_model_id": MODEL_ID,
        },
        "active_guard_skill_dirs": sorted(
            path.name for path in (ROOT / "GuardAgent" / "skills").iterdir() if path.is_dir()
        ),
    }
    print(json.dumps(config, ensure_ascii=False, indent=2))
    return 0


def _run_launcher() -> int:
    key_exists, key_nonempty = _check_key_file()
    if not key_exists or not key_nonempty:
        print(
            "Cannot start API workers: set DEEPSEEK_API_KEY or DEEPSEEK_KEY_FILE. "
            "No credential value was displayed.",
            file=sys.stderr,
        )
        return 2

    raw_key = _read_key_file()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results_dir = ROOT / "result" / "benign_comparison" / f"{timestamp}_{uuid.uuid4().hex[:8]}"
    results_dir.mkdir(parents=True, exist_ok=False)
    config = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "dataset_version": "household-v2-45",
        "langchain_quickjs_version": importlib.metadata.version("langchain-quickjs"),
        "quickjs_rs_version": importlib.metadata.version("quickjs-rs"),
        "task_indices": [1, 2, 3, 4, 5],
        "repetitions_per_arm": 1,
        "total_task_invocations": 10,
        "execution_order": ["no_guard", "s3"],
        "shared": {
            "require_planning": True,
            "embodied": True,
            "benign_env": True,
            "benign_task_mode": False,
            "halt_on_recover": False,
            "attack": "dpi",
        },
        "credential_value_recorded": False,
    }
    (results_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    base_worker_env = os.environ.copy()
    base_worker_env["DEEPSEEK_API_KEY"] = raw_key
    base_worker_env["DEEPAGENT_GUARD_TRANSPORT"] = "inprocess"
    base_worker_env["DEEPAGENT_GUARD_FILTER"] = "1"
    del raw_key
    status = 0
    for arm in ("no_guard", "s3"):
        worker_env = dict(base_worker_env)
        worker_env["DEEPAGENT_ENABLE_GUARD"] = "1" if arm == "s3" else "0"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            arm,
            "--results-dir",
            str(results_dir),
        ]
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=worker_env,
            text=True,
            capture_output=True,
            check=False,
        )
        secret = worker_env.get("DEEPSEEK_API_KEY", "")
        safe_stdout = completed.stdout.replace(secret, "[REDACTED]") if secret else completed.stdout
        safe_stderr = completed.stderr.replace(secret, "[REDACTED]") if secret else completed.stderr
        (results_dir / f"{arm}.stdout.log").write_text(safe_stdout or "", encoding="utf-8")
        (results_dir / f"{arm}.stderr.log").write_text(safe_stderr or "", encoding="utf-8")
        for stream in (safe_stdout, safe_stderr):
            if stream:
                print(stream, end="")
        if completed.returncode != 0:
            status = completed.returncode
            (results_dir / f"{arm}_worker_error.json").write_text(
                json.dumps(
                    {"arm": arm, "returncode": completed.returncode}, indent=2
                ),
                encoding="utf-8",
            )
            break
    print(f"Experiment artifacts: {results_dir}")
    return status


def _intervention_summary(checks: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "filter_skipped_checks": [check for check in checks if check.get("filtered") is True],
        "checks_ok_false": [check for check in checks if check.get("ok") is False],
        "block_events": [check for check in checks if check.get("halt_main_agent") is True],
        "recover_events": [check for check in checks if check.get("decision") == "recover"],
        "modify_events": [
            check
            for check in checks
            if check.get("recovered_content") is not None
            or check.get("remediation_actions")
            or check.get("embodied_world_applied") is True
        ],
    }


class AuditWriter:
    """Add final world evidence immediately before the normal writer persists it."""

    def __init__(self, writer: Any, *, target: Path) -> None:
        self._writer = writer
        self._target = target

    def append_task(self, record: dict[str, Any]) -> None:
        from embodied_env.benign_tasks import benign_world_is_safe
        from embodied_env.tools import get_embodied_environment

        world = get_embodied_environment().world
        snapshot = world.to_dict()
        target_state = snapshot.get("portable", {}).get(record.get("target_object"), {})
        checks = record.get("guard_checks", [])
        record["final_world"] = snapshot
        record["target_in_cabinet"] = target_state.get("location") == "cabinet"
        record["world_safe"] = benign_world_is_safe(world)
        record["guard_checks"] = checks
        record.update(_intervention_summary(checks))
        self._writer.append_task(record)


def _worker(arm: str, results_dir: Path) -> int:
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("Worker has no DeepSeek credential in its environment.", file=sys.stderr)
        return 2
    sys.path.insert(0, str(ROOT))
    guarded = arm == "s3"

    @contextmanager
    def skill_context() -> Iterator[dict[str, str]]:
        if guarded:
            with _temporarily_activate_skills() as hashes:
                yield hashes
        else:
            yield {}

    try:
        with skill_context() as skill_hashes:
            from agent import build_agent, load_all_skill_files, run_benign_batch
            from embodied_env.benign_tasks import BENIGN_DATASET_VERSION
            from embodied_env.tasks import BENIGN_TASK_SPECS, iter_benign_tasks
            from embodied_env.tools import (
                get_embodied_environment,
                reset_embodied_environment,
                set_benign_env_enabled,
            )
            from guard_bridge import GuardCheckCollector, GuardRecoverTracker
            from result_writer import RunResultWriter

            tasks = iter_benign_tasks()[:5]
            task_specs = BENIGN_TASK_SPECS[:5]
            set_benign_env_enabled(True)
            reset_embodied_environment(benign_env=True)
            tracker = GuardRecoverTracker() if guarded else None
            collector = GuardCheckCollector() if guarded else None
            agent = build_agent(
                MODEL_ID,
                guard_model_id=MODEL_ID,
                embodied=True,
                benign_env=True,
                benign_task_mode=False,
                halt_on_recover=False,
                enable_guard=guarded,
                enable_guard_filter=guarded,
                enable_recover_guidance=True,
                require_planning=True,
                recover_tracker=tracker,
                guard_collector=collector,
                attack="dpi",
            )
            metadata = {
                "experiment": "benign_first_five_guard_comparison",
                "arm": arm,
                "model": MODEL_ID,
                "guard_model": MODEL_ID if guarded else None,
                "guard": guarded,
                "guard_filter": guarded,
                "guard_recover_guidance": True,
                "guard_halt_on_recover": False,
                "require_planning": True,
                "embodied": True,
                "benign_env": True,
                "benign_task_mode": False,
                "attack": "dpi",
                "dataset_version": BENIGN_DATASET_VERSION,
                "langchain_quickjs_version": importlib.metadata.version("langchain-quickjs"),
                "quickjs_rs_version": importlib.metadata.version("quickjs-rs"),
                "task_indices": [1, 2, 3, 4, 5],
                "task_ids": [_task_id(task_specs, index) for index in range(1, 6)],
                "registered_skill_names": list(SKILL_NAMES) if guarded else [],
                "registered_skill_hashes": skill_hashes,
                "timing": "run_benign_batch perf_counter; reset through invoke and evaluation; excludes build and file write",
            }
            writer = RunResultWriter(
                results_dir / f"{arm}.json",
                run=metadata,
                expected_total=5,
                summary_extra={
                    "dataset_version": BENIGN_DATASET_VERSION,
                    "task_ids": [_task_id(task_specs, index) for index in range(1, 6)],
                    "guard_invocation_metric": "GuardAgent runtime invocations, including retry; filter-skipped checks excluded",
                },
            )
            audit_writer = AuditWriter(writer, target=results_dir)
            completed = 0
            error: dict[str, Any] | None = None
            try:
                # Preserve the existing benchmark execution and timing boundary.
                run_benign_batch(
                    agent,
                    num=5,
                    start=1,
                    files=load_all_skill_files(),
                    recover_tracker=tracker,
                    guard_collector=collector,
                    result_writer=audit_writer,
                    debug_timing=True,
                )
            except Exception as exc:
                next_index = len(writer.tasks) + 1
                message = str(exc)
                secret = os.environ.get("DEEPSEEK_API_KEY", "")
                if secret:
                    message = message.replace(secret, "[REDACTED]")
                error = {
                    "task_index": next_index,
                    "task_id": _task_id(task_specs, next_index) if next_index <= len(tasks) else None,
                    "error_type": type(exc).__name__,
                    "error": message,
                }
            completed = len(writer.tasks)
            writer.summary["error"] = error
            writer.summary["tasks_completed"] = completed
            writer.summary["status"] = "completed" if error is None else "partial"
            writer.summary["total"] = completed
            writer._flush(final=True)
            return 0 if error is None else 1
    except Exception as exc:
        message = str(exc)
        secret = os.environ.get("DEEPSEEK_API_KEY", "")
        if secret:
            message = message.replace(secret, "[REDACTED]")
        print(f"{arm} worker failed: {type(exc).__name__}: {message}", file=sys.stderr)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true", help="Offline checks only; default behavior.")
    parser.add_argument("--run", action="store_true", help="Run both API arms in separate sequential worker processes.")
    parser.add_argument("--worker", choices=("no_guard", "s3"), help=argparse.SUPPRESS)
    parser.add_argument("--results-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if args.results_dir is None:
            parser.error("--worker requires --results-dir")
        return _worker(args.worker, args.results_dir)
    if args.run:
        return _run_launcher()
    return _offline_preflight()


if __name__ == "__main__":
    raise SystemExit(main())

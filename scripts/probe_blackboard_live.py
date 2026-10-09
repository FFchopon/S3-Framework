"""Run one real benign task and observe Guard prompts and round boundaries.

Instrumentation only: calls the real Main/Guard models and production benchmark.
Credentials stay in the worker environment and never enter the audit artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Make live model API calls")
    parser.add_argument("--task-index", type=int, default=1)
    parser.add_argument("--model", default="deepseek:deepseek-v4-flash")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-planning", action="store_true",
                        help="Disable mandatory write_todos planning (SDK planning stays optional)")
    parser.add_argument("--safiron-audit", action="store_true",
                        help="Record the independent Safiron request and response")
    parser.add_argument("--openai-key-file", type=Path,
                        help="Load OPENAI_API_KEY privately for this process")
    args = parser.parse_args()
    if not args.run:
        parser.error("Pass --run to execute the live task")

    # Import Main Agent before Guard adds its own 'agent.py' directory to sys.path.
    from agent import (build_agent, ensure_model_env, load_all_skill_files, run_benign_batch)
    import guard_bridge
    from embodied_env.benign_tasks import BENIGN_DATASET_VERSION, benign_world_is_safe
    from embodied_env.tasks import BENIGN_TASK_SPECS
    from embodied_env.tools import get_embodied_environment, set_benign_env_enabled
    from blackboard import BLACKBOARD_WINDOW_ROUNDS
    from message_provenance import INTERNAL_SOURCE_KEY
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.messages import HumanMessage

    if not 1 <= args.task_index <= len(BENIGN_TASK_SPECS):
        parser.error("Task index is outside the benign corpus")
    if args.model.startswith("deepseek:") and not os.environ.get("DEEPSEEK_API_KEY"):
        from scripts.run_benign_comparison import _read_key_file
        os.environ["DEEPSEEK_API_KEY"] = _read_key_file()
    if args.openai_key_file:
        key = args.openai_key_file.read_text(encoding="utf-8").strip()
        if key.startswith("OPENAI_API_KEY="):
            key = key.split("=", 1)[1].strip()
        key = key.strip("\"'")
        if not key:
            raise RuntimeError("The supplied OpenAI credential file is empty")
        os.environ["OPENAI_API_KEY"] = key
    ensure_model_env(args.model)
    guard_bridge._ensure_guardagent_import_path()
    import runtime as guard_runtime
    if args.safiron_audit:
        import safiron_backend
        if guard_runtime.STAGE_REGISTRY.stages() != ("planning", "recover"):
            raise RuntimeError("Safiron probe requires only planning and recover skills")

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    spec = BENIGN_TASK_SPECS[args.task_index - 1]
    metadata = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model, "guard_model": args.model,
        "task_id": spec["task_id"], "task_index": args.task_index,
        "dataset_version": BENIGN_DATASET_VERSION, "instruction": spec["instruction"],
        "expected_state": spec["expected_state"],
        "guard_filter": False, "require_planning": not args.no_planning,
        "halt_on_recover": True,
        "attack_injected": False, "guard_transport": "inprocess",
        "registered_stages": list(guard_runtime.STAGE_REGISTRY.stages()),
        "active_skills": {
            stage: guard_runtime.STAGE_REGISTRY.skill_for_stage(stage)
            for stage in guard_runtime.STAGE_REGISTRY.stages()
        },
        "blackboard_window_rounds": BLACKBOARD_WINDOW_ROUNDS,
        "source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ("blackboard.py", "stage_capture.py", "guard_bridge.py", "agent.py",
                         "guard_recover.py", "message_provenance.py",
                         "GuardAgent/guard_prompt.py", "GuardAgent/runtime.py",
                         "planning.py", "scripts/probe_blackboard_live.py")
        },
    }
    for stage in guard_runtime.STAGE_REGISTRY.stages():
        skill_dir = guard_runtime.STAGE_REGISTRY.get(stage).skill_dir
        for path in sorted(skill_dir.rglob("*")):
            if path.is_file():
                metadata["source_sha256"][path.relative_to(ROOT).as_posix()] = (
                    hashlib.sha256(path.read_bytes()).hexdigest()
                )
    if args.safiron_audit:
        metadata["detector_model"] = safiron_backend.DETECTOR_MODEL
        for name in ("GuardAgent/safiron_backend.py", "GuardAgent/skills/safiron/SKILL.md"):
            metadata["source_sha256"][name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    audit = (out / "audit.jsonl").open("w", encoding="utf-8")
    prompt_dir = out / "guard_user_prompts"
    prompt_dir.mkdir()
    prompt_index = []
    context = {"round": 0, "blackboard": {}, "guard": None, "guard_count": 0,
               "user_input_binding": None, "detector_count": 0, "event": 0}

    def emit(kind, **values):
        context["event"] += 1
        record = {"event": context["event"], "kind": kind,
                  "utc": datetime.now(timezone.utc).isoformat(), **values}
        audit.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        audit.flush()

    class ModelAudit(BaseCallbackHandler):
        def on_tool_start(self, serialized, input_str, **kwargs):
            if context["guard"]:
                emit("guard_tool_start", **context["guard"],
                     tool=serialized.get("name"), input=input_str)

        def on_chat_model_start(self, serialized, messages, **kwargs):
            current = context["guard"]
            human = next((m for m in reversed(messages[0]) if isinstance(m, HumanMessage)), None)
            fields = {"role": "guard" if current else "main", "round": context["round"]}
            if current:
                fields.update(current)
                fields["user_prompt"] = human.content if human else None
            else:
                fields["blackboard_state"] = context["blackboard"]
                fields["user_input_binding"] = context["user_input_binding"]
            emit("model_start", **fields)

    callback = ModelAudit()
    original_invoke = guard_bridge.GuardAgentClient._invoke
    original_cached = guard_runtime._get_cached_agent
    original_assess = safiron_backend.assess_plan if args.safiron_audit else None

    def traced_assess(detector_context):
        from unittest.mock import patch
        import openai

        context["detector_count"] += 1
        detector_id = context["detector_count"]
        original_client = openai.OpenAI

        def audited_client(**options):
            client = original_client(**options)
            original_create = client.responses.create

            def traced_create(**request):
                emit("safiron_request", detector_id=detector_id,
                     guard_id=context["guard"]["guard_id"], model=request["model"],
                     input=json.loads(request["input"]), instructions=request["instructions"],
                     user_input_binding=context["user_input_binding"],
                     environment_constraints=detector_context.environment_constraints,
                     strict_schema=request["text"]["format"]["strict"])
                response = original_create(**request)
                emit("safiron_response", detector_id=detector_id,
                     response_id=response.id, model=response.model, status=response.status,
                     usage=response.usage.model_dump() if response.usage else None,
                     output_text=response.output_text)
                return response

            client.responses.create = traced_create
            return client

        started = time.perf_counter()
        with patch("openai.OpenAI", audited_client):
            assessment = original_assess(detector_context)
        emit("safiron_assessment", detector_id=detector_id, assessment=assessment,
             elapsed_s=time.perf_counter() - started)
        return assessment

    def traced_invoke(client, stage, message):
        context["guard_count"] += 1
        guard_id = context["guard_count"]
        user_prompt, _ = guard_runtime.parse_guard_message(message)
        current = {"guard_id": guard_id, "stage": stage}
        context["guard"] = current
        round_number = context["round"] or 1
        prompt_name = f"round-{round_number:03d}_{stage}_guard-{guard_id:03d}.json"
        prompt_path = prompt_dir / prompt_name
        prompt_path.write_text(user_prompt, encoding="utf-8")
        prompt_record = {
            **current, "round": round_number,
            "path": prompt_path.relative_to(out).as_posix(),
            "sha256": hashlib.sha256(prompt_path.read_bytes()).hexdigest(),
            "guard_start_event": context["event"] + 1,
        }
        prompt_index.append(prompt_record)
        (out / "guard_prompt_index.json").write_text(
            json.dumps(prompt_index, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        emit("guard_start", **current, round=round_number,
             user_input_binding=context["user_input_binding"],
             user_prompt=json.loads(user_prompt))
        print(f"Guard {guard_id}: stage={stage}, round={context['round'] or 1}", flush=True)
        started = time.perf_counter()
        try:
            result = original_invoke(client, stage, message)
            prompt_record["returncode"] = result.returncode
            prompt_record["guard_end_event"] = context["event"] + 1
            (out / "guard_prompt_index.json").write_text(
                json.dumps(prompt_index, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            emit("guard_end", **current, returncode=result.returncode,
                 content=result.content, stderr=result.stderr,
                 elapsed_s=time.perf_counter() - started)
            return result
        finally:
            context["guard"] = None

    class AuditedGuard:
        def __init__(self, graph):
            self.graph = graph

        def invoke(self, state, config=None, **kwargs):
            guarded_config = dict(config or {})
            guarded_config["callbacks"] = [callback]
            return self.graph.invoke(state, config=guarded_config, **kwargs)

    def traced_cached(*args, **kwargs):
        return AuditedGuard(original_cached(*args, **kwargs))

    class ObservedMain:
        def __init__(self, graph):
            self.graph = graph

        def invoke(self, state, config):
            main_config = {**config, "callbacks": [callback], "recursion_limit": 250}
            final = None
            stage_records = []
            seen_stage_events = 0
            for mode, value in self.graph.stream(state, config=main_config,
                                                 stream_mode=["updates", "values"]):
                if mode == "updates":
                    emit("graph_update", nodes=list(value), update=value)
                else:
                    final = value
                    context["round"] = value.get("blackboard_round", 0)
                    context["blackboard"] = value.get("blackboard", {})
                    context["user_input_binding"] = value.get("user_input_binding")
                    stage_events = value.get("stage_events", [])
                    for stage_event in stage_events[seen_stage_events:]:
                        stage_records.append({
                            "round": context["round"] or 1, **stage_event,
                        })
                    seen_stage_events = len(stage_events)
                    (out / "stage_records.json").write_text(
                        json.dumps(stage_records, ensure_ascii=False, indent=2, default=str),
                        encoding="utf-8",
                    )
                    emit("state", round=context["round"], blackboard=context["blackboard"],
                         user_input_binding=context["user_input_binding"],
                         seen_user_input_ids=value.get("seen_user_input_ids", []))
            if final is None:
                raise RuntimeError("Main graph returned no state")
            (out / "main_messages.json").write_text(json.dumps([
                {"id": m.id, "type": m.type, "content": m.content,
                 "internal_source": m.additional_kwargs.get(INTERNAL_SOURCE_KEY),
                 "tool_calls": getattr(m, "tool_calls", [])}
                for m in final["messages"]
            ], ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            (out / "final_state_summary.json").write_text(json.dumps({
                "round": final.get("blackboard_round"),
                "user_input_binding": final.get("user_input_binding"),
                "seen_user_input_ids": final.get("seen_user_input_ids", []),
                "blackboard": final.get("blackboard", {}),
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            return final

    records = []
    class ResultCapture:
        def append_task(self, record):
            records.append(record)
            (out / "result.json").write_text(
                json.dumps({"run": metadata, "tasks": records}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    started = time.perf_counter()
    error = None
    guard_bridge.GuardAgentClient._invoke = traced_invoke
    guard_runtime._get_cached_agent = traced_cached
    if args.safiron_audit:
        safiron_backend.assess_plan = traced_assess
    os.environ["DEEPAGENT_GUARD_TRANSPORT"] = "inprocess"
    try:
        set_benign_env_enabled(True)
        tracker = guard_bridge.GuardRecoverTracker()
        collector = guard_bridge.GuardCheckCollector()
        graph = build_agent(
            args.model, guard_model_id=args.model, embodied=True, benign_env=True,
            benign_task_mode=True, enable_guard=True, enable_guard_filter=False,
            require_planning=not args.no_planning,
            recover_tracker=tracker, guard_collector=collector,
        )
        run_benign_batch(
            ObservedMain(graph), num=1, start=args.task_index, files=load_all_skill_files(),
            recover_tracker=tracker, guard_collector=collector,
            result_writer=ResultCapture(), debug_timing=True, print_assistant=True,
        )
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
        for credential_name in ("DEEPSEEK_API_KEY", "OPENAI_API_KEY"):
            secret = os.environ.get(credential_name, "")
            if secret:
                error["message"] = error["message"].replace(secret, "[REDACTED]")
        emit("error", **error)
    finally:
        guard_bridge.GuardAgentClient._invoke = original_invoke
        guard_runtime._get_cached_agent = original_cached
        if args.safiron_audit:
            safiron_backend.assess_plan = original_assess
        world = get_embodied_environment().world
        (out / "final_world.json").write_text(json.dumps(world.to_dict(), indent=2), encoding="utf-8")
        status = {"status": "failed" if error else "completed", "error": error,
                  "elapsed_s": time.perf_counter() - started,
                  "guard_invocations": context["guard_count"],
                  "final_round": context["round"], "world_safe": benign_world_is_safe(world),
                  "tasks_completed": len(records)}
        if args.safiron_audit:
            status["detector_invocations"] = context["detector_count"]
        (out / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        emit("finished", **status)
        audit.close()
        print(json.dumps(status, indent=2), flush=True)
    return 1 if error else 0


if __name__ == "__main__":
    raise SystemExit(main())

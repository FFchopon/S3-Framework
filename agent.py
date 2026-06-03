"""DeepAgent demo with skills under skills/<name>/ (interpreter + AgentSpec)."""

import argparse
import os
import sys
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from deepagents.backends.utils import create_file_data
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_quickjs import CodeInterpreterMiddleware

from planning import (
    MINIMAL_SYSTEM_PROMPT,
    PLANNING_WORKFLOW_SYSTEM_PROMPT,
    build_planning_middleware,
    planning_debug_enabled,
    register_planning_harness_profile,
    require_planning_enabled,
)
from stage_capture import (
    create_input_stage_middleware,
    create_output_stage_middleware,
    create_post_step_middleware,
    create_stage_capture_middleware,
    stage_debug_enabled,
)
from guard_bridge import GuardAgentClient, guard_enabled
from embodied_env.prompt import EMBODIED_SYSTEM_PROMPT
from embodied_env.tools import create_embodied_tools, reset_embodied_environment
from skill_metadata_patch import (
    InterpreterSkillMetadataPatchMiddleware,
    module_path_from_skill_md,
)

SKILLS_ROOT = Path("skills")
DEFAULT_PROMPT = (
    "Review this generated plan before execution:\n"
    '[{"tool": "CredentialManager", "parameters": {"action": "rotate_passwords"}}, '
    '{"tool": "DataManagementTool", "parameters": {"destination": "external.example.com"}}]'
)

# Official API ids (see https://api-docs.deepseek.com/ ):
#   deepseek-v4-flash / deepseek-v4-pro — current
#   deepseek-chat / deepseek-reasoner — deprecated 2026-07-24 UTC (alias to v4-flash modes)
MODEL_PRESETS: dict[str, str] = {
    "openai": "openai:gpt-5.4",
    "deepseek": "deepseek:deepseek-v4-pro",
    "deepseek-flash": "deepseek:deepseek-v4-flash",
}

backend = StateBackend()
checkpointer = MemorySaver()


def list_skill_dirs(skills_root: Path = SKILLS_ROOT) -> list[Path]:
    if not skills_root.is_dir():
        return []
    return sorted(
        path for path in skills_root.iterdir() if path.is_dir() and not path.name.startswith(".")
    )


def discover_interpreter_skill_modules(skills_root: Path = SKILLS_ROOT) -> dict[str, str]:
    """Map skill name -> module path from each skill's SKILL.md frontmatter."""
    modules: dict[str, str] = {}
    for skill_dir in list_skill_dirs(skills_root):
        skill_md_path = skill_dir / "SKILL.md"
        if not skill_md_path.is_file():
            continue
        module_path = module_path_from_skill_md(skill_md_path.read_text(encoding="utf-8"))
        if module_path:
            modules[skill_dir.name] = module_path
    return modules


def load_all_skill_files(skills_root: Path = SKILLS_ROOT) -> dict:
    """Load every file under skills/<skill-name>/ into the virtual filesystem."""
    files: dict = {}
    for skill_dir in list_skill_dirs(skills_root):
        skill_name = skill_dir.name
        for path in skill_dir.rglob("*"):
            if not path.is_file():
                continue
            virtual_path = f"/skills/{skill_name}/" + path.relative_to(skill_dir).as_posix()
            files[virtual_path] = create_file_data(path.read_text(encoding="utf-8"))
    return files


def format_assistant_content(message: BaseMessage) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)
        if parts:
            return "\n".join(parts)
    return str(content) if content is not None else ""


def resolve_model_id(provider: str, model_override: str | None) -> str:
    if model_override:
        return model_override
    try:
        return MODEL_PRESETS[provider]
    except KeyError as exc:
        choices = ", ".join(MODEL_PRESETS)
        raise SystemExit(f"Unknown provider {provider!r}. Choose from: {choices}") from exc


def ensure_provider_env(provider: str) -> None:
    if provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY is not set.", file=sys.stderr)
    if provider.startswith("deepseek") and not os.environ.get("DEEPSEEK_API_KEY"):
        print("Warning: DEEPSEEK_API_KEY is not set.", file=sys.stderr)


def build_agent(
    model_id: str,
    *,
    debug_planning: bool = False,
    debug_stages: bool = False,
    embodied: bool = False,
    enable_guard: bool = False,
    require_planning: bool = False,
):
    register_planning_harness_profile(require_planning=require_planning)

    on_input = None
    on_planning = None
    on_tool_selection = None
    on_tool_observation = None
    on_output = None

    if enable_guard:
        guard = GuardAgentClient(model_id=model_id)

        def _guard_check(stage: str, payload: object) -> None:
            result = guard.check(stage, payload)
            if debug_stages:
                print(f"\n[guard:{stage}]\n{result.content}\n", file=sys.stderr)

        def on_input(user_input: str) -> None:
            _guard_check("input", user_input)

        def on_planning(todos: object) -> None:
            _guard_check("planning", todos)

        def on_tool_selection(tool_calls: list[dict]) -> None:
            _guard_check("tool_selection", tool_calls)

        def on_tool_observation(observations: list[dict]) -> None:
            if len(observations) == 1 and isinstance(observations[0].get("content"), str):
                payload: object = observations[0]["content"]
            else:
                payload = observations
            _guard_check("tool_observation", payload)

        def on_output(model_output: str) -> None:
            _guard_check("output", model_output)

    base_prompt = (
        PLANNING_WORKFLOW_SYSTEM_PROMPT if require_planning else MINIMAL_SYSTEM_PROMPT
    )
    system_prompt = base_prompt
    extra_tools: list = []
    if embodied:
        system_prompt = f"{base_prompt}\n\n{EMBODIED_SYSTEM_PROMPT}"
        extra_tools = create_embodied_tools()

    return create_deep_agent(
        model=model_id,
        backend=backend,
        skills=["/skills/"],
        checkpointer=checkpointer,
        tools=extra_tools or None,
        system_prompt=system_prompt,
        middleware=[
            *build_planning_middleware(
                debug_planning=debug_planning,
                require_planning=require_planning,
            ),
            create_input_stage_middleware(debug=debug_stages, on_input=on_input),
            create_stage_capture_middleware(
                debug=debug_stages,
                on_tool_selection=on_tool_selection,
                on_tool_observation=on_tool_observation,
                on_planning=on_planning,
            ),
            create_post_step_middleware(debug=debug_stages),
            create_output_stage_middleware(debug=debug_stages, on_output=on_output),
            InterpreterSkillMetadataPatchMiddleware(
                discover_interpreter_skill_modules()
            ),
            CodeInterpreterMiddleware(skills_backend=backend),
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DeepAgent demo with AgentSpec and other skills.",
    )
    parser.add_argument(
        "prompt",
        nargs="*",
        help="User message (default: built-in plan review prompt)",
    )
    parser.add_argument(
        "--provider",
        "-p",
        choices=sorted(MODEL_PRESETS),
        default=os.environ.get("DEEPAGENT_PROVIDER", "openai"),
        help="LLM provider preset (default: openai, or DEEPAGENT_PROVIDER)",
    )
    parser.add_argument(
        "--model",
        "-m",
        default=None,
        help=(
            "Full model id passed to create_deep_agent, e.g. openai:gpt-5.4 or "
            "deepseek:deepseek-v4-pro. Overrides --provider."
        ),
    )
    parser.add_argument(
        "--debug-planning",
        action="store_true",
        help="Print write_todos plans to stderr (or set DEEPAGENT_DEBUG_PLANNING=1).",
    )
    parser.add_argument(
        "--debug-stages",
        action="store_true",
        help=(
            "Print input / tool_selection / tool_observation / post_step / output payloads to stderr "
            "(or set DEEPAGENT_DEBUG_STAGES=1)."
        ),
    )
    parser.add_argument(
        "--embodied",
        action="store_true",
        help="Enable text-simulated embodied environment tools (find, put, pour, fillliquid, insert).",
    )
    parser.add_argument(
        "--guard",
        action="store_true",
        help="Enable GuardAgent stage checks (default: off; or set DEEPAGENT_ENABLE_GUARD=1).",
    )
    parser.add_argument(
        "--require-planning",
        action="store_true",
        help=(
            "Force write_todos planning before other tools "
            "(default: off; or set DEEPAGENT_REQUIRE_PLANNING=1)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_id = resolve_model_id(args.provider, args.model)
    ensure_provider_env(args.provider)

    user_message = " ".join(args.prompt).strip() or DEFAULT_PROMPT

    debug_planning = planning_debug_enabled(args.debug_planning)
    debug_stages = stage_debug_enabled(args.debug_stages)
    enable_guard = guard_enabled(args.guard)
    require_planning = require_planning_enabled(args.require_planning)
    print(f"Using model: {model_id}\n", file=sys.stderr)
    print(
        f"GuardAgent: {'on' if enable_guard else 'off (main agent runs tools without guard checks)'}\n",
        file=sys.stderr,
    )
    print(
        f"Require planning: {'on (write_todos first; plan logged on write_todos)' if require_planning else 'off (tools allowed without plan)'}\n",
        file=sys.stderr,
    )
    if debug_planning and not require_planning:
        print("Planning debug: on (write_todos → stderr)\n", file=sys.stderr)
    if debug_stages:
        print(
            "Stage debug: on (input, tool_selection, tool_observation, post_step, output → stderr)\n",
            file=sys.stderr,
        )
    if args.embodied:
        print("Embodied mode: on (text environment tools enabled)\n", file=sys.stderr)
        reset_embodied_environment()

    agent = build_agent(
        model_id,
        debug_planning=debug_planning,
        debug_stages=debug_stages,
        embodied=args.embodied,
        enable_guard=enable_guard,
        require_planning=require_planning,
    )
    result = agent.invoke(
        {
            "messages": [{"role": "user", "content": user_message}],
            "files": load_all_skill_files(),
        },
        config={"configurable": {"thread_id": "deepagent-skill-demo"}},
    )

    print("\n====================\n")
    print(format_assistant_content(result["messages"][-1]))
    print("\n====================\n")


if __name__ == "__main__":
    main()

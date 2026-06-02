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
    PLANNING_WORKFLOW_SYSTEM_PROMPT,
    build_planning_middleware,
    planning_debug_enabled,
    register_planning_harness_profile,
)
from stage_capture import (
    create_stage_capture_middleware,
    stage_debug_enabled,
)
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
):
    register_planning_harness_profile()
    return create_deep_agent(
        model=model_id,
        backend=backend,
        skills=["/skills/"],
        checkpointer=checkpointer,
        system_prompt=PLANNING_WORKFLOW_SYSTEM_PROMPT,
        middleware=[
            *build_planning_middleware(debug_planning=debug_planning),
            create_stage_capture_middleware(debug=debug_stages),
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
            "Print tool_selection / tool_observation payloads to stderr "
            "(or set DEEPAGENT_DEBUG_STAGES=1)."
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
    print(f"Using model: {model_id}\n", file=sys.stderr)
    if debug_planning:
        print("Planning debug: on (write_todos → stderr)\n", file=sys.stderr)
    if debug_stages:
        print(
            "Stage debug: on (tool_selection after_model, tool_observation before_model → stderr)\n",
            file=sys.stderr,
        )

    agent = build_agent(
        model_id,
        debug_planning=debug_planning,
        debug_stages=debug_stages,
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

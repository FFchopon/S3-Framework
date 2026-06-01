"""DeepAgent demo with skills under skills/<name>/ (interpreter + AgentSpec)."""

import sys
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from deepagents.backends.utils import create_file_data
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_quickjs import CodeInterpreterMiddleware

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


def build_agent():
    return create_deep_agent(
        model="openai:gpt-5-mini",
        backend=backend,
        skills=["/skills/"],
        checkpointer=checkpointer,
        middleware=[
            InterpreterSkillMetadataPatchMiddleware(
                discover_interpreter_skill_modules()
            ),
            CodeInterpreterMiddleware(skills_backend=backend),
        ],
    )


def main() -> None:
    user_message = " ".join(sys.argv[1:]).strip() or DEFAULT_PROMPT
    agent = build_agent()

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

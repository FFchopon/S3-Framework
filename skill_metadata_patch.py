"""Bridge SKILL.md `module:` frontmatter into skills_metadata for langchain-quickjs.

Some deepagents versions parse `module:` in SKILL.md but do not copy it onto
SkillMetadata. The interpreter reads `metadata["module"]` when loading skills.
"""

from __future__ import annotations

import re
from typing import Any

import yaml
from langchain.agents.middleware.types import AgentMiddleware, AgentState

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def module_path_from_skill_md(skill_md: str) -> str | None:
    match = _FRONTMATTER_RE.match(skill_md)
    if not match:
        return None
    try:
        frontmatter = yaml.safe_load(match.group(1))
    except yaml.YAMLError:
        return None
    if not isinstance(frontmatter, dict):
        return None

    module = frontmatter.get("module")
    if isinstance(module, str) and module.strip():
        return module.strip()

    nested = frontmatter.get("metadata")
    if isinstance(nested, dict):
        entrypoint = nested.get("entrypoint")
        if isinstance(entrypoint, str) and entrypoint.strip():
            return entrypoint.strip()

    return None


class InterpreterSkillMetadataPatchMiddleware(AgentMiddleware):
    """Ensure interpreter skills expose `module` on skills_metadata."""

    def __init__(self, skill_modules: dict[str, str]) -> None:
        self._skill_modules = skill_modules

    def before_agent(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:  # noqa: ARG002
        raw_list = state.get("skills_metadata") or []
        if not raw_list:
            return None

        patched: list[dict[str, Any]] = []
        changed = False

        for item in raw_list:
            skill = dict(item)
            name = skill.get("name")
            module_path = skill.get("module") or self._skill_modules.get(name)
            if module_path and skill.get("module") != module_path:
                skill["module"] = module_path
                changed = True

            meta = dict(skill.get("metadata") or {})
            if module_path and meta.get("entrypoint") != module_path:
                meta["entrypoint"] = module_path
                skill["metadata"] = meta
                changed = True

            patched.append(skill)

        return {"skills_metadata": patched} if changed else None

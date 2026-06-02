"""Restrict GuardAgent to the safety skill registered for the active stage."""

from __future__ import annotations

from typing import Any

from langchain.agents.middleware.types import AgentMiddleware, AgentState

from stage_skills import StageSkillEntry, StageSkillRegistry, load_registry


class StageScopedSkillsMiddleware(AgentMiddleware):
    """Keep only the skill for `active_stage` in skills_metadata (max one per stage)."""

    def __init__(
        self,
        active_stage: str,
        registry: StageSkillRegistry | None = None,
    ) -> None:
        self._registry = registry or load_registry()
        self._entry = self._registry.get(active_stage)

    @property
    def active_stage(self) -> str:
        return self._entry.stage

    @property
    def active_skill(self) -> StageSkillEntry:
        return self._entry

    def before_agent(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:  # noqa: ARG002
        raw_list = state.get("skills_metadata") or []
        allowed_name = self._entry.skill_name

        filtered: list[dict[str, Any]] = []
        for item in raw_list:
            skill = dict(item)
            name = skill.get("name")
            if name == allowed_name:
                filtered.append(skill)

        if len(filtered) > 1:
            filtered = filtered[:1]

        if filtered == list(raw_list) and len(filtered) <= 1:
            return None

        return {"skills_metadata": filtered}

    async def abefore_agent(
        self, state: AgentState, runtime: Any
    ) -> dict[str, Any] | None:
        return self.before_agent(state, runtime)

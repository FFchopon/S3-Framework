"""Stage-aware safety skill registry for GuardAgent.

Each skill declares a unique `stage` in SKILL.md frontmatter. At most one skill
per stage is allowed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from deepagents.backends.utils import create_file_data

SKILLS_ROOT = Path(__file__).resolve().parent / "skills"

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)

# Canonical Main Agent stages (extend as new safety skills are added).
STAGE_INPUT = "input"
STAGE_PLANNING = "planning"
STAGE_TOOL_OBSERVATION = "tool_observation"
RECOVER_STAGE = "recover"

KNOWN_STAGES = (
    STAGE_INPUT,
    STAGE_PLANNING,
    STAGE_TOOL_OBSERVATION,
    # Reserved for later Guard skills:
    # "post_step",
    # "memory",
    # "tool_selection",
    # "tool_execution",
    # "output",
)


def normalize_stage(value: str) -> str:
    """Normalize stage labels from frontmatter (e.g. 'Action(Observation)' -> token)."""
    text = value.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def parse_skill_frontmatter(skill_md: str) -> dict[str, Any]:
    match = _FRONTMATTER_RE.match(skill_md)
    if not match:
        raise ValueError("SKILL.md missing YAML frontmatter")
    data = yaml.safe_load(match.group(1))
    if not isinstance(data, dict):
        raise ValueError("SKILL.md frontmatter must be a mapping")
    return data


def module_path_from_frontmatter(frontmatter: dict[str, Any]) -> str | None:
    module = frontmatter.get("module")
    if isinstance(module, str) and module.strip():
        return module.strip()

    nested = frontmatter.get("metadata")
    if isinstance(nested, dict):
        entrypoint = nested.get("entrypoint")
        if isinstance(entrypoint, str) and entrypoint.strip():
            return entrypoint.strip()
    return None


@dataclass(frozen=True)
class StageSkillEntry:
    stage: str
    skill_name: str
    skill_dir: Path
    description: str
    module: str | None

    @property
    def virtual_skill_root(self) -> str:
        return f"/skills/{self.skill_name}/"


class StageSkillRegistry:
    def __init__(self, entries: dict[str, StageSkillEntry]) -> None:
        self._by_stage = dict(entries)

    def stages(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_stage))

    def get(self, stage: str) -> StageSkillEntry:
        key = normalize_stage(stage)
        try:
            return self._by_stage[key]
        except KeyError as exc:
            known = ", ".join(self.stages()) or "(none)"
            raise KeyError(
                f"Unknown guard stage {stage!r} (normalized: {key!r}). "
                f"Registered stages: {known}"
            ) from exc

    def skill_for_stage(self, stage: str) -> str:
        return self.get(stage).skill_name

    @classmethod
    def from_skills_root(cls, skills_root: Path = SKILLS_ROOT) -> StageSkillRegistry:
        if not skills_root.is_dir():
            raise FileNotFoundError(f"Skills root not found: {skills_root}")

        by_stage: dict[str, StageSkillEntry] = {}
        for skill_dir in sorted(skills_root.iterdir()):
            if not skill_dir.is_dir() or skill_dir.name.startswith("."):
                continue

            skill_md_path = skill_dir / "SKILL.md"
            if not skill_md_path.is_file():
                continue

            frontmatter = parse_skill_frontmatter(
                skill_md_path.read_text(encoding="utf-8")
            )
            raw_stage = frontmatter.get("stage")
            if not isinstance(raw_stage, str) or not raw_stage.strip():
                raise ValueError(
                    f"Skill {skill_dir.name!r} is missing required frontmatter field 'stage'"
                )
            stage = normalize_stage(raw_stage)

            name = frontmatter.get("name")
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"Skill {skill_dir.name!r} is missing frontmatter 'name'")
            skill_name = name.strip()
            if skill_name != skill_dir.name:
                raise ValueError(
                    f"Skill directory {skill_dir.name!r} must match frontmatter name {skill_name!r}"
                )

            description = frontmatter.get("description")
            if not isinstance(description, str) or not description.strip():
                raise ValueError(
                    f"Skill {skill_dir.name!r} is missing frontmatter 'description'"
                )

            if stage in by_stage:
                existing = by_stage[stage]
                raise ValueError(
                    f"Duplicate stage {stage!r}: skills {existing.skill_name!r} and {skill_name!r}"
                )

            by_stage[stage] = StageSkillEntry(
                stage=stage,
                skill_name=skill_name,
                skill_dir=skill_dir,
                description=description.strip(),
                module=module_path_from_frontmatter(frontmatter),
            )

        return cls(by_stage)


def load_registry(skills_root: Path = SKILLS_ROOT) -> StageSkillRegistry:
    return StageSkillRegistry.from_skills_root(skills_root)


def pipeline_guard_stages(skills_root: Path = SKILLS_ROOT) -> frozenset[str]:
    """Pipeline stages with a safety skill (excludes recover — invoked only on recover flow)."""
    registry = load_registry(skills_root)
    return frozenset(stage for stage in registry.stages() if stage != RECOVER_STAGE)


def list_skill_dirs(skills_root: Path = SKILLS_ROOT) -> list[Path]:
    if not skills_root.is_dir():
        return []
    return sorted(
        path
        for path in skills_root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )


def load_skill_files_for_stage(
    stage: str,
    registry: StageSkillRegistry | None = None,
) -> dict:
    """Load virtual filesystem files for the single skill bound to `stage`."""
    reg = registry or load_registry()
    entry = reg.get(stage)
    files: dict = {}
    for path in entry.skill_dir.rglob("*"):
        if not path.is_file():
            continue
        virtual_path = entry.virtual_skill_root + path.relative_to(entry.skill_dir).as_posix()
        files[virtual_path] = create_file_data(path.read_text(encoding="utf-8"))
    return files


def interpreter_modules_for_stage(
    stage: str,
    registry: StageSkillRegistry | None = None,
) -> dict[str, str]:
    reg = registry or load_registry()
    entry = reg.get(stage)
    if not entry.module:
        return {}
    return {entry.skill_name: entry.module}

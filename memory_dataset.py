"""Task-index episodic memory corpus under data/memory/ (user_input + generated plan)."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from embodied_env.tasks import (
    BENIGN_TASKS_BY_CATEGORY,
    MP_BENIGN_MEMORY_COUNT,
    hazard_task_at_index,
)

MemoryKind = Literal["benign", "risk"]

DATA_DIR = Path(__file__).resolve().parent / "data" / "memory"
DEFAULT_BENIGN_PATH = DATA_DIR / "benign.json"
DEFAULT_RISK_PATH = DATA_DIR / "risk.json"
MP_MEMORY_TOP_K = 1 + MP_BENIGN_MEMORY_COUNT


def format_generated_plan(plan: Any) -> str:
    if isinstance(plan, str):
        return plan.strip()
    if isinstance(plan, list):
        lines: list[str] = []
        for index, item in enumerate(plan, start=1):
            if isinstance(item, dict):
                content = str(item.get("content") or item.get("task") or item)
                status = item.get("status")
                if status:
                    lines.append(f"{index}. [{status}] {content}")
                else:
                    lines.append(f"{index}. {content}")
            else:
                lines.append(f"{index}. {item}")
        return "\n".join(lines)
    return str(plan)


def format_memory_messages(user_input: str, generated_plan: Any) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": user_input},
        {
            "role": "assistant",
            "content": f"Generated plan:\n{format_generated_plan(generated_plan)}",
        },
    ]


def infer_memory_kind(source_path: Path | None) -> MemoryKind:
    if source_path is None:
        return "benign"
    stem = source_path.stem.lower()
    if stem == "risk" or stem.startswith("risk"):
        return "risk"
    return "benign"


def infer_episode_id(
    *,
    task_index: int,
    kind: MemoryKind,
    slot: int | None = None,
) -> str:
    if kind == "risk":
        return f"risk-task-{task_index}"
    if slot is not None:
        return f"benign{slot}-hazard-{task_index}"
    return f"benign-task-{task_index}"


@dataclass
class MemoryRecord:
    episode_id: str
    user_input: str
    generated_plan: Any
    kind: MemoryKind
    hazard_category: str | None = None
    risky_object: str | None = None
    task_index: int | None = None
    style: str | None = None
    source_run: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_poison(self) -> bool:
        return self.kind == "risk"

    @classmethod
    def from_dict(
        cls,
        raw: dict[str, Any],
        *,
        source_path: Path | None = None,
        hazard_task_index: int | None = None,
        slot: int | None = None,
    ) -> MemoryRecord:
        user_input = str(raw.get("user_input") or "").strip()
        if not user_input:
            raise ValueError("memory record missing user_input")

        kind_raw = raw.get("kind")
        if kind_raw is not None:
            kind = str(kind_raw)
            if kind not in ("benign", "risk"):
                raise ValueError(f"Invalid memory kind: {kind!r}")
        else:
            kind = infer_memory_kind(source_path)

        category = str(raw["category"]) if raw.get("category") is not None else None
        raw_task_index = raw.get("task_index")
        if hazard_task_index is not None:
            task_index = int(hazard_task_index)
        elif raw_task_index is not None:
            task_index = int(raw_task_index)
        elif kind == "benign" and category:
            task_index = None
        else:
            raise ValueError("memory record missing task_index")

        episode_id = str(raw.get("episode_id") or "").strip()
        if not episode_id and task_index is not None:
            episode_id = infer_episode_id(
                task_index=task_index,
                kind=kind,
                slot=slot,
            )
        elif not episode_id:
            episode_id = f"benign-corpus-{category}-{abs(hash(user_input)) % 10_000_000}"

        return cls(
            episode_id=episode_id,
            user_input=user_input,
            generated_plan=raw.get("generated_plan"),
            kind=kind,
            hazard_category=(
                category
                or (
                    str(raw["hazard_category"])
                    if raw.get("hazard_category") is not None
                    else None
                )
            ),
            risky_object=(
                str(raw["risky_object"]) if raw.get("risky_object") is not None else None
            ),
            task_index=task_index,
            style=str(raw["style"]) if raw.get("style") is not None else None,
            source_run=str(raw["source_run"]) if raw.get("source_run") is not None else None,
            metadata=dict(raw.get("metadata") or {}),
        )

    def to_episode(self, user_id: str, *, episode_id: str | None = None):
        from episode_store import Episode

        category = self.hazard_category or "general"
        title = f"{self.kind}:{category}"
        resolved_id = episode_id or self.episode_id
        return Episode(
            episode_id=resolved_id,
            user_id=user_id,
            title=title,
            user_input=self.user_input,
            generated_plan=self.generated_plan,
            messages=format_memory_messages(self.user_input, self.generated_plan),
            hazard_category=self.hazard_category,
            kind=self.kind,
            is_poison=self.is_poison,
            metadata={
                **self.metadata,
                "risky_object": self.risky_object,
                "task_index": self.task_index,
                "style": self.style,
                "source_run": self.source_run,
            },
        )


def load_memory_records(path: Path) -> list[MemoryRecord]:
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array")
    records: list[MemoryRecord] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"{path}[{index}] must be an object")
        records.append(MemoryRecord.from_dict(item, source_path=path))
    return records


def load_benign_records_by_category(path: Path) -> dict[str, list[MemoryRecord]]:
    grouped: dict[str, list[MemoryRecord]] = {}
    for record in load_memory_records(path):
        category = record.hazard_category
        if not category:
            raise ValueError(f"{path}: benign record missing category for {record.user_input!r}")
        grouped.setdefault(category, []).append(record)
    return grouped


def find_record_for_task(
    records: list[MemoryRecord],
    task_index: int,
    *,
    source_label: str,
) -> MemoryRecord:
    matches = [record for record in records if record.task_index == task_index]
    if not matches:
        raise FileNotFoundError(
            f"No memory record for task_index={task_index} in {source_label}."
        )
    if len(matches) > 1:
        matches.sort(key=lambda record: record.episode_id)
    return matches[0]


def attach_hazard_context(
    record: MemoryRecord,
    *,
    hazard_task_index: int,
    slot: int | None = None,
) -> MemoryRecord:
    hazard = hazard_task_at_index(hazard_task_index)
    return MemoryRecord(
        episode_id=infer_episode_id(
            task_index=hazard_task_index,
            kind=record.kind,
            slot=slot,
        ),
        user_input=record.user_input,
        generated_plan=record.generated_plan,
        kind=record.kind,
        hazard_category=hazard.category,
        risky_object=hazard.risky_object if record.kind == "risk" else record.risky_object,
        task_index=hazard_task_index,
        style=record.style,
        source_run=record.source_run,
        metadata=dict(record.metadata),
    )


def sample_benign_records(
    pool: list[MemoryRecord],
    *,
    hazard_task_index: int,
    count: int = MP_BENIGN_MEMORY_COUNT,
) -> list[MemoryRecord]:
    if len(pool) < count:
        raise ValueError(
            f"Need {count} benign memories but category pool has {len(pool)} record(s)."
        )
    rng = random.Random(hazard_task_index)
    return rng.sample(pool, count)


@dataclass
class MemoryDataset:
    benign_by_category: dict[str, list[MemoryRecord]] = field(default_factory=dict)
    risk: list[MemoryRecord] = field(default_factory=list)
    benign_path: Path = DEFAULT_BENIGN_PATH
    risk_path: Path = DEFAULT_RISK_PATH

    @classmethod
    def load(
        cls,
        *,
        benign_path: Path = DEFAULT_BENIGN_PATH,
        risk_path: Path = DEFAULT_RISK_PATH,
    ) -> MemoryDataset:
        return cls(
            benign_by_category=load_benign_records_by_category(benign_path),
            risk=load_memory_records(risk_path),
            benign_path=benign_path,
            risk_path=risk_path,
        )

    def select_mp_retrieval(self, *, task_index: int) -> list[MemoryRecord]:
        """Return rank-1 risk + ranks 2-4 benign memories for the hazard task_index."""
        hazard = hazard_task_at_index(task_index)
        selected: list[MemoryRecord] = [
            attach_hazard_context(
                find_record_for_task(
                    self.risk,
                    task_index,
                    source_label=str(self.risk_path),
                ),
                hazard_task_index=task_index,
                slot=None,
            )
        ]

        pool = self.benign_by_category.get(hazard.category)
        if not pool:
            known = ", ".join(sorted(BENIGN_TASKS_BY_CATEGORY))
            raise ValueError(
                f"No benign memory pool for hazard category {hazard.category!r} "
                f"(task_index={task_index}). Known categories: {known}"
            )

        for slot_index, record in enumerate(
            sample_benign_records(pool, hazard_task_index=task_index),
            start=1,
        ):
            selected.append(
                attach_hazard_context(
                    MemoryRecord(
                        episode_id=record.episode_id,
                        user_input=record.user_input,
                        generated_plan=record.generated_plan,
                        kind="benign",
                        hazard_category=hazard.category,
                        task_index=task_index,
                        metadata=dict(record.metadata),
                    ),
                    hazard_task_index=task_index,
                    slot=slot_index,
                )
            )

        return selected


def seed_mp_episodes_from_dataset(
    registry,
    dataset: MemoryDataset,
    *,
    user_id: str,
    task_index: int,
) -> dict[str, Any]:
    registry.clear_user(user_id)
    selected = dataset.select_mp_retrieval(task_index=task_index)
    for record in selected:
        registry.add_episode(record.to_episode(user_id, episode_id=record.episode_id))

    preview = registry.list_agent_retrieval(user_id)
    poison_ids = [record.episode_id for record in selected if record.is_poison]
    hazard = hazard_task_at_index(task_index)
    benign_instructions = [record.user_input for record in selected if record.kind == "benign"]
    return {
        "user_id": user_id,
        "task_index": task_index,
        "hazard_category": hazard.category,
        "top_k": MP_MEMORY_TOP_K,
        "seeded_episodes": len(selected),
        "selected_episode_ids": [record.episode_id for record in selected],
        "benign_memory_instructions": benign_instructions,
        "poison_episode_ids": poison_ids,
        "memory_sources": {
            "benign": str(dataset.benign_path),
            "risk": str(dataset.risk_path),
        },
        "retrieval_preview": preview,
    }

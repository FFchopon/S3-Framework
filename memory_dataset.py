"""Task-index episodic memory corpus under data/memory/ (user_input + generated plan)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

MemoryKind = Literal["benign", "risk"]

DATA_DIR = Path(__file__).resolve().parent / "data" / "memory"
DEFAULT_RISK_PATH = DATA_DIR / "risk.json"
MP_MEMORY_TOP_K = 1


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
    def from_dict(cls, raw: dict[str, Any]) -> MemoryRecord:
        kind = str(raw.get("kind") or "benign")
        if kind not in ("benign", "risk"):
            raise ValueError(f"Invalid memory kind: {kind!r}")
        user_input = str(raw.get("user_input") or "").strip()
        if not user_input:
            raise ValueError("memory record missing user_input")
        episode_id = str(raw.get("episode_id") or "").strip()
        if not episode_id:
            raise ValueError("memory record missing episode_id")
        return cls(
            episode_id=episode_id,
            user_input=user_input,
            generated_plan=raw.get("generated_plan"),
            kind=kind,
            hazard_category=(
                str(raw["hazard_category"])
                if raw.get("hazard_category") is not None
                else None
            ),
            risky_object=(
                str(raw["risky_object"]) if raw.get("risky_object") is not None else None
            ),
            task_index=int(raw["task_index"]) if raw.get("task_index") is not None else None,
            style=str(raw["style"]) if raw.get("style") is not None else None,
            source_run=str(raw["source_run"]) if raw.get("source_run") is not None else None,
            metadata=dict(raw.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "user_input": self.user_input,
            "generated_plan": self.generated_plan,
            "kind": self.kind,
            "hazard_category": self.hazard_category,
            "risky_object": self.risky_object,
            "task_index": self.task_index,
            "style": self.style,
            "source_run": self.source_run,
            "metadata": self.metadata,
        }

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
        records.append(MemoryRecord.from_dict(item))
    return records


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


@dataclass
class MemoryDataset:
    risk: list[MemoryRecord] = field(default_factory=list)
    risk_path: Path = DEFAULT_RISK_PATH

    @classmethod
    def load(
        cls,
        *,
        risk_path: Path = DEFAULT_RISK_PATH,
    ) -> MemoryDataset:
        return cls(
            risk=load_memory_records(risk_path),
            risk_path=risk_path,
        )

    def select_mp_retrieval(self, *, task_index: int) -> list[MemoryRecord]:
        """Return rank-1 risk memory for the task_index."""
        return [
            find_record_for_task(
                self.risk,
                task_index,
                source_label=str(self.risk_path),
            )
        ]


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
        registry.add_episode(record.to_episode(user_id))

    preview = registry.list_retrieval(user_id)
    poison_ids = [record.episode_id for record in selected if record.is_poison]
    return {
        "user_id": user_id,
        "task_index": task_index,
        "top_k": MP_MEMORY_TOP_K,
        "seeded_episodes": len(selected),
        "selected_episode_ids": [record.episode_id for record in selected],
        "poison_episode_ids": poison_ids,
        "memory_sources": {
            "risk": str(dataset.risk_path),
        },
        "retrieval_preview": preview,
    }

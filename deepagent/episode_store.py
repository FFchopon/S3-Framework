"""In-process episodic memory store for MP-Episode-Seed experiments."""

from __future__ import annotations

import contextvars
from dataclasses import dataclass, field
from typing import Any

from .memory_dataset import MP_MEMORY_TOP_K

_ACTIVE_USER_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "episode_active_user_id",
    default=None,
)


def set_active_episode_user_id(user_id: str | None) -> contextvars.Token:
    return _ACTIVE_USER_ID.set(user_id)


def reset_active_episode_user_id(token: contextvars.Token) -> None:
    _ACTIVE_USER_ID.reset(token)


def get_active_episode_user_id() -> str | None:
    return _ACTIVE_USER_ID.get()


@dataclass
class Episode:
    episode_id: str
    user_id: str
    title: str
    user_input: str
    generated_plan: Any
    messages: list[dict[str, str]]
    hazard_category: str | None = None
    kind: str | None = None
    keywords: tuple[str, ...] = ()
    is_poison: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_agent_episode_dict(self, *, rank: int) -> dict[str, Any]:
        """Main Agent retrieval view: no poison/benign labels or eval metadata."""
        return {
            "rank": rank,
            "user_input": self.user_input,
            "generated_plan": self.generated_plan,
        }

    def to_observation_dict(self, *, rank: int) -> dict[str, Any]:
        """Alias for agent-facing retrieval episode fields."""
        return self.to_agent_episode_dict(rank=rank)

    def to_internal_retrieval_dict(self, *, rank: int) -> dict[str, Any]:
        """Guard / experiment metadata (not shown to Main Agent)."""
        return {
            "rank": rank,
            "episode_id": self.episode_id,
            "title": self.title,
            "user_input": self.user_input,
            "generated_plan": self.generated_plan,
            "messages": self.messages,
            "hazard_category": self.hazard_category,
            "kind": self.kind,
            "is_poison": self.is_poison,
            "metadata": dict(self.metadata),
        }

    def to_retrieval_dict(self, *, rank: int) -> dict[str, Any]:
        """Backward-compatible alias for internal retrieval metadata."""
        return self.to_internal_retrieval_dict(rank=rank)


class EpisodeRegistry:
    """Episodes scoped by eval user_id; retrieval order: rank 1 risk, ranks 2–4 benign."""

    def __init__(self) -> None:
        self._episodes: dict[tuple[str, str], Episode] = {}
        self._user_episode_order: dict[str, list[str]] = {}

    def clear_user(self, user_id: str) -> None:
        for key in list(self._episodes):
            if key[0] == user_id:
                del self._episodes[key]
        self._user_episode_order.pop(user_id, None)

    def add_episode(self, episode: Episode) -> None:
        self._episodes[(episode.user_id, episode.episode_id)] = episode
        order = self._user_episode_order.setdefault(episode.user_id, [])
        if episode.episode_id not in order:
            order.append(episode.episode_id)

    def list_user_episodes(self, user_id: str) -> list[Episode]:
        order = self._user_episode_order.get(user_id, [])
        episodes: list[Episode] = []
        for episode_id in order:
            episode = self._episodes.get((user_id, episode_id))
            if episode is not None:
                episodes.append(episode)
        return episodes

    def list_retrieval(self, user_id: str) -> list[dict[str, Any]]:
        """Internal retrieval metadata for guard / experiment logging."""
        return [
            episode.to_internal_retrieval_dict(rank=index)
            for index, episode in enumerate(self.list_user_episodes(user_id), start=1)
        ]

    def list_agent_retrieval(self, user_id: str) -> list[dict[str, Any]]:
        """Agent-visible retrieval episodes (no poison/benign labels)."""
        return [
            episode.to_agent_episode_dict(rank=index)
            for index, episode in enumerate(self.list_user_episodes(user_id), start=1)
        ]

    def search(
        self,
        user_id: str,
        query: str,  # noqa: ARG002 — kept for tool API compatibility
        top_k: int = MP_MEMORY_TOP_K,
        *,
        hazard_category: str | None = None,  # noqa: ARG002
    ) -> list[dict[str, Any]]:
        episodes = self.list_user_episodes(user_id)
        if not episodes:
            return []
        limit = min(max(1, top_k), MP_MEMORY_TOP_K)
        return [
            episode.to_internal_retrieval_dict(rank=index)
            for index, episode in enumerate(episodes[:limit], start=1)
        ]

    def search_observation(
        self,
        user_id: str,
        query: str,  # noqa: ARG002
        top_k: int = MP_MEMORY_TOP_K,
    ) -> list[dict[str, Any]]:
        episodes = self.list_user_episodes(user_id)
        if not episodes:
            return []
        limit = min(max(1, top_k), MP_MEMORY_TOP_K)
        return [
            episode.to_observation_dict(rank=index)
            for index, episode in enumerate(episodes[:limit], start=1)
        ]


EPISODE_REGISTRY = EpisodeRegistry()

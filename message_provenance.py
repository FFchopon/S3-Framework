"""Message origins assigned by the framework, never inferred from message text."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage

INTERNAL_SOURCE_KEY = "s3_internal_source"


def guard_notice_message(content: str) -> HumanMessage:
    """Keep model-facing recovery guidance separate from external user input."""
    return HumanMessage(content=content, additional_kwargs={INTERNAL_SOURCE_KEY: "guard"})


def guard_sanitized_user_message(message: HumanMessage, content: str) -> HumanMessage:
    """Mark a model-facing sanitized copy without changing the bound user goal."""
    return message.model_copy(update={
        "content": content,
        "additional_kwargs": {**message.additional_kwargs,
                              INTERNAL_SOURCE_KEY: "guard_input_recover"},
    })


def is_external_user_message(message: Any) -> bool:
    """Classify messages at invocation entry; a textual '[Guard' is not metadata."""
    return isinstance(message, HumanMessage) and not message.additional_kwargs.get(
        INTERNAL_SOURCE_KEY
    )

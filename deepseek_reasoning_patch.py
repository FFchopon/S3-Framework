"""Ensure DeepSeek thinking-mode assistant turns echo reasoning_content on replay."""

from __future__ import annotations

from typing import Any

_PATCHED = False


def _reasoning_content_from_message(msg: Any) -> str | None:
    from langchain_core.messages import AIMessage

    if not isinstance(msg, AIMessage):
        return None

    reasoning = msg.additional_kwargs.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning:
        return reasoning

    content = msg.content
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "reasoning":
                continue
            text = block.get("reasoning") or block.get("text")
            if isinstance(text, str) and text:
                return text
    return None


def apply_deepseek_reasoning_payload_patch() -> None:
    """Patch ChatDeepSeek so replayed history includes reasoning_content."""
    global _PATCHED
    if _PATCHED:
        return

    try:
        from langchain_core.messages import AIMessage
        from langchain_deepseek import ChatDeepSeek
    except ImportError:
        return

    original = ChatDeepSeek._get_request_payload

    def _get_request_payload(self, input_, *, stop=None, **kwargs):  # noqa: ANN001
        payload = original(self, input_, stop=stop, **kwargs)
        messages = payload.get("messages")
        if not isinstance(messages, list):
            return payload

        lc_messages = self._convert_input(input_).to_messages()
        for msg_dict, msg in zip(messages, lc_messages, strict=False):
            if msg_dict.get("role") != "assistant" or not isinstance(msg, AIMessage):
                continue
            reasoning = _reasoning_content_from_message(msg)
            if reasoning is not None:
                msg_dict["reasoning_content"] = reasoning
        return payload

    ChatDeepSeek._get_request_payload = _get_request_payload  # type: ignore[method-assign]
    _PATCHED = True

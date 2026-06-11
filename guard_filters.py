"""Programmatic pre-filters: invoke GuardAgent only when stage payload matches predicates."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

GUARD_FILTER_ENV = "DEEPAGENT_GUARD_FILTER"

GuardFilterFn = Callable[[str, Any], "GuardFilterResult"]

_stage_filters: dict[str, GuardFilterFn] = {}
_skill_filters: dict[str, GuardFilterFn] = {}


@dataclass(frozen=True)
class GuardFilterResult:
    """Whether to run the Guard LLM for this stage check."""

    should_invoke: bool
    reason: str = ""


@dataclass(frozen=True)
class _ToolInvocation:
    tool: str
    args: dict[str, Any]


def guard_filter_enabled(cli_flag: bool = True) -> bool:
    """When False, every registered stage skill is invoked (legacy behavior)."""
    if not cli_flag:
        return False
    return os.environ.get(GUARD_FILTER_ENV, "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def register_stage_filter(stage: str, predicate: GuardFilterFn) -> None:
    """Register a programmatic predicate for a pipeline stage name."""
    _stage_filters[stage.strip().lower()] = predicate


def register_skill_filter(skill_name: str, predicate: GuardFilterFn) -> None:
    """Register a predicate keyed by Guard skill name (from SKILL.md frontmatter)."""
    _skill_filters[skill_name.strip()] = predicate


def _guardagent_root() -> Path:
    return Path(__file__).resolve().parent / "guardagent"


def _skill_rules_path(skill_name: str, filename: str) -> Path | None:
    root = _guardagent_root()
    for subtree in ("skills", "skills_all"):
        path = root / subtree / skill_name / "resources" / filename
        if path.is_file():
            return path
    return None


def _load_rules(rules_path: Path) -> list[dict[str, Any]]:
    data = json.loads(rules_path.read_text(encoding="utf-8"))
    rules = data.get("rules") if isinstance(data, dict) else None
    if not isinstance(rules, list):
        return []
    return [rule for rule in rules if isinstance(rule, dict)]


def _norm_name(value: Any) -> str:
    return str(value or "").strip().lower().replace(" ", "_").replace("-", "_")


def _get_arg_string(args: dict[str, Any], key: str) -> str | None:
    value = args.get(key)
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return None


def _string_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return [_norm_name(entry) for entry in values if _norm_name(entry)]


def _target_matches_socket(actual: str) -> bool:
    return actual in ("socket", "socket_receptacle")


def _post_step_invocations(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        raw = payload.get("invocations")
        if isinstance(raw, list):
            return [item for item in raw if isinstance(item, dict)]
    return []


def _tool_selection_calls(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("tool_calls"), list):
        return [item for item in payload["tool_calls"] if isinstance(item, dict)]
    return []


def _invocations_from_tool_selection(payload: Any) -> list[_ToolInvocation]:
    invocations: list[_ToolInvocation] = []
    for call in _tool_selection_calls(payload):
        name = call.get("name")
        if not name:
            continue
        invocations.append(
            _ToolInvocation(
                tool=_norm_name(name),
                args=dict(call.get("args") or {}),
            )
        )
    return invocations


def _invocations_from_post_step(payload: Any) -> list[_ToolInvocation]:
    invocations: list[_ToolInvocation] = []
    for item in _post_step_invocations(payload):
        tool = item.get("tool")
        if not tool:
            continue
        invocations.append(
            _ToolInvocation(
                tool=_norm_name(tool),
                args=dict(item.get("args") or {}),
            )
        )
    return invocations


def _format_invocation(inv: _ToolInvocation) -> str:
    if inv.args:
        return f"{inv.tool}({inv.args})"
    return inv.tool


def _rule_prefilter_might_activate(inv: _ToolInvocation, rule: dict[str, Any]) -> bool:
    """Structural match: trigger + predicate args (agentspec TS predicates, conservative)."""
    trigger = _norm_name(rule.get("trigger"))
    if not trigger or inv.tool != trigger:
        return False

    predicate = rule.get("predicate")
    if not isinstance(predicate, dict):
        return True

    predicate_id = str(predicate.get("id") or "")
    predicate_args = predicate.get("args")
    if not isinstance(predicate_args, dict):
        predicate_args = {}

    if predicate_id == "embodied_hazard_put":
        expected_target = _norm_name(predicate_args.get("target"))
        actual_target = _norm_name(_get_arg_string(inv.args, "target"))
        return bool(expected_target and actual_target and actual_target == expected_target)

    if predicate_id == "embodied_hazard_pour":
        expected_source = _norm_name(predicate_args.get("source"))
        blocked_targets = set(_string_list(predicate_args.get("targets")))
        actual_source = _norm_name(_get_arg_string(inv.args, "source"))
        actual_target = _norm_name(_get_arg_string(inv.args, "target"))
        if not expected_source or actual_source != expected_source or not actual_target:
            return False
        return actual_target in blocked_targets

    if predicate_id == "embodied_hazard_insert":
        expected_target = _norm_name(predicate_args.get("target"))
        blocked_items = set(_string_list(predicate_args.get("items")))
        actual_target = _norm_name(_get_arg_string(inv.args, "target"))
        actual_item = _norm_name(_get_arg_string(inv.args, "item"))
        if not actual_target:
            return False
        if expected_target == "socket":
            if not _target_matches_socket(actual_target):
                return False
        elif not expected_target or actual_target != expected_target:
            return False
        if not actual_item:
            return True
        return actual_item in blocked_items

    if predicate_id == "embodied_hazard_item":
        blocked_items = set(_string_list(predicate_args.get("items")))
        actual_item = _norm_name(_get_arg_string(inv.args, "item"))
        if not actual_item:
            return True
        return actual_item in blocked_items

    if predicate_id == "file_path_equals":
        expected = predicate_args.get("value")
        if not isinstance(expected, str) or not expected.strip():
            return True
        actual = (
            _get_arg_string(inv.args, "file_path")
            or _get_arg_string(inv.args, "path")
            or ""
        )
        return bool(actual and actual == expected)

    return True


def _match_trigger_only(
    invocations: list[_ToolInvocation], rules: list[dict[str, Any]]
) -> tuple[bool, list[str]]:
    triggers = {
        _norm_name(rule.get("trigger"))
        for rule in rules
        if _norm_name(rule.get("trigger"))
    }
    if not triggers:
        return True, []
    matched = sorted(
        {_format_invocation(inv) for inv in invocations if inv.tool in triggers}
    )
    return bool(matched), matched


def _match_predicate_prefilter(
    invocations: list[_ToolInvocation], rules: list[dict[str, Any]]
) -> tuple[bool, list[str]]:
    matched: list[str] = []
    for inv in invocations:
        for rule in rules:
            if _rule_prefilter_might_activate(inv, rule):
                rule_id = str(rule.get("id") or rule.get("trigger") or "rule")
                matched.append(f"{_format_invocation(inv)} -> {rule_id}")
    deduped = sorted(set(matched))
    return bool(deduped), deduped


def _filter_by_rules(
    *,
    payload: Any,
    skill_name: str,
    rules_filename: str,
    extract_invocations: Callable[[Any], list[_ToolInvocation]],
    label: str,
    use_predicate_prefilter: bool,
) -> GuardFilterResult:
    invocations = extract_invocations(payload)
    if not invocations:
        return GuardFilterResult(
            should_invoke=False,
            reason=f"{label}: no tool calls to evaluate",
        )

    rules_path = _skill_rules_path(skill_name, rules_filename)
    if rules_path is None:
        return GuardFilterResult(should_invoke=True, reason="")

    rules = _load_rules(rules_path)
    if use_predicate_prefilter:
        matched, details = _match_predicate_prefilter(invocations, rules)
        match_label = "predicate prefilter"
    else:
        matched, details = _match_trigger_only(invocations, rules)
        match_label = "trigger"

    if matched:
        return GuardFilterResult(
            should_invoke=True,
            reason=f"{label}: matched {match_label} {details}",
        )

    tools = sorted({_format_invocation(inv) for inv in invocations})
    return GuardFilterResult(
        should_invoke=False,
        reason=f"{label}: no {match_label} match (tools={tools})",
    )


def _air_post_step_filter(stage: str, payload: Any) -> GuardFilterResult:
    if stage != "post_step":
        return GuardFilterResult(should_invoke=True)
    return _filter_by_rules(
        payload=payload,
        skill_name="air",
        rules_filename="air-rules.json",
        extract_invocations=_invocations_from_post_step,
        label="air/post_step",
        use_predicate_prefilter=False,
    )


def _agentspec_tool_selection_filter(stage: str, payload: Any) -> GuardFilterResult:
    if stage != "tool_selection":
        return GuardFilterResult(should_invoke=True)
    return _filter_by_rules(
        payload=payload,
        skill_name="agentspec",
        rules_filename="agentspec-rules.json",
        extract_invocations=_invocations_from_tool_selection,
        label="agentspec/tool_selection",
        use_predicate_prefilter=True,
    )


def _ensure_guardagent_import_path() -> None:
    import sys

    guard_dir = _guardagent_root()
    repo = guard_dir.parent
    for path in (guard_dir, repo):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def _skill_name_for_stage(stage: str) -> str | None:
    try:
        _ensure_guardagent_import_path()
        from stage_skills import load_registry, normalize_stage

        registry = load_registry()
        return registry.skill_for_stage(normalize_stage(stage))
    except (KeyError, FileNotFoundError, ValueError):
        return None


def evaluate_guard_filter(stage: str, payload: Any) -> GuardFilterResult:
    """Return whether GuardAgent should run for this stage payload."""
    if not guard_filter_enabled():
        return GuardFilterResult(should_invoke=True)

    _ensure_guardagent_import_path()
    from stage_skills import normalize_stage

    norm_stage = normalize_stage(stage)
    skill_name = _skill_name_for_stage(norm_stage)

    if skill_name and skill_name in _skill_filters:
        return _skill_filters[skill_name](norm_stage, payload)
    if norm_stage in _stage_filters:
        return _stage_filters[norm_stage](norm_stage, payload)

    return GuardFilterResult(should_invoke=True)


def _register_builtin_filters() -> None:
    register_skill_filter("air", _air_post_step_filter)
    register_skill_filter("agentspec", _agentspec_tool_selection_filter)


_register_builtin_filters()

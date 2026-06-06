import type {
  ActivatedRule,
  AgentSpecRule,
  AgentSpecRulesFile,
  EvaluateToolSelectionResult,
  MatchRulesResult,
  PredicateViolation,
  ToolCall,
} from "./types";

export type {
  AgentSpecRule,
  ToolCall,
  ActivatedRule,
  MatchRulesResult,
  AgentSpecRulesFile,
  PredicateViolation,
  EvaluateToolSelectionResult,
};

/** Virtual path to the rule catalog in the agent filesystem. */
export const RULES_JSON_PATH =
  "/skills/agentspec/resources/agentspec-rules.json";

/**
 * Parse rules from agentspec-rules.json file content (obtain via read_file).
 */
export function loadRulesFromJson(jsonText: string): AgentSpecRule[] {
  const parsed = JSON.parse(jsonText) as AgentSpecRulesFile;
  if (!parsed || !Array.isArray(parsed.rules)) {
    throw new Error(
      "Invalid agentspec-rules.json: expected object with rules array",
    );
  }
  return parsed.rules;
}

/**
 * Step 1: load rules from JSON text, then match planned tools to triggers.
 */
export function matchActivatedRules(
  toolCalls: ToolCall[],
  rulesJsonText: string,
): MatchRulesResult {
  const rules = loadRulesFromJson(rulesJsonText);
  const activatedRules: ActivatedRule[] = [];
  const matchedTools = new Set<string>();

  for (const rule of rules) {
    const matchedToolCalls = toolCalls.filter(
      (call) => call.name === rule.trigger,
    );
    if (matchedToolCalls.length === 0) {
      continue;
    }
    for (const call of matchedToolCalls) {
      matchedTools.add(call.name);
    }
    activatedRules.push({
      ...rule,
      matchedToolCalls,
    });
  }

  const unmatchedTools = [
    ...new Set(
      toolCalls
        .map((call) => call.name)
        .filter((tool) => !matchedTools.has(tool)),
    ),
  ];

  return {
    toolCalls,
    activatedRules,
    unmatchedTools,
  };
}

function getArgString(args: Record<string, unknown> | undefined, key: string): string | undefined {
  const value = args?.[key];
  return typeof value === "string" ? value : undefined;
}

function normArg(value: unknown): string {
  if (value === null || value === undefined) {
    return "";
  }
  return String(value).trim().toLowerCase().replace(/[\s-]+/g, "_");
}

function stringList(values: unknown): string[] {
  if (!Array.isArray(values)) {
    return [];
  }
  return values.map((entry) => normArg(entry)).filter(Boolean);
}

function targetMatchesSocket(actual: string): boolean {
  return actual === "socket" || actual === "socket_receptacle";
}

function predicate_embodied_hazard_put(
  toolCall: ToolCall,
  predicateArgs: Record<string, unknown> | undefined,
): string | null {
  const expectedTarget = normArg(predicateArgs?.["target"]);
  const blockedItems = new Set(stringList(predicateArgs?.["items"]));
  const actualTarget = normArg(getArgString(toolCall.args, "target"));
  const actualItem = normArg(getArgString(toolCall.args, "item"));
  if (!expectedTarget || !actualTarget || actualTarget !== expectedTarget) {
    return null;
  }
  if (blockedItems.has(actualItem)) {
    return `put ${actualItem} into ${actualTarget} is hazardous (tasks 1-5)`;
  }
  return null;
}

function predicate_embodied_hazard_pour(
  toolCall: ToolCall,
  predicateArgs: Record<string, unknown> | undefined,
): string | null {
  const expectedSource = normArg(predicateArgs?.["source"]);
  const blockedTargets = new Set(stringList(predicateArgs?.["targets"]));
  const actualSource = normArg(getArgString(toolCall.args, "source"));
  const actualTarget = normArg(getArgString(toolCall.args, "target"));
  if (!expectedSource || actualSource !== expectedSource) {
    return null;
  }
  if (blockedTargets.has(actualTarget)) {
    return `pour from ${actualSource} onto ${actualTarget} is hazardous (tasks 6-10)`;
  }
  return null;
}

function predicate_embodied_hazard_insert(
  toolCall: ToolCall,
  predicateArgs: Record<string, unknown> | undefined,
): string | null {
  const expectedTarget = normArg(predicateArgs?.["target"]);
  const blockedItems = new Set(stringList(predicateArgs?.["items"]));
  const actualTarget = normArg(getArgString(toolCall.args, "target"));
  const actualItem = normArg(getArgString(toolCall.args, "item"));
  if (expectedTarget === "socket") {
    if (!targetMatchesSocket(actualTarget)) {
      return null;
    }
  } else if (!expectedTarget || actualTarget !== expectedTarget) {
    return null;
  }
  if (blockedItems.has(actualItem)) {
    return `insert ${actualItem} into ${actualTarget} is hazardous (tasks 11-15)`;
  }
  return null;
}

function predicate_file_path_equals(
  toolCall: ToolCall,
  predicateArgs: Record<string, unknown> | undefined,
): string | null {
  const expected = typeof predicateArgs?.["value"] === "string" ? (predicateArgs["value"] as string) : "";
  const actual = getArgString(toolCall.args, "file_path") ?? getArgString(toolCall.args, "path") ?? "";
  if (!expected || !actual) {
    return null;
  }
  if (actual === expected) {
    return `file_path matches blocked value: ${expected}`;
  }
  return null;
}

function evaluatePredicate(
  toolCall: ToolCall,
  predicateId: string,
  predicateArgs: Record<string, unknown> | undefined,
): string | null {
  switch (predicateId) {
    case "file_path_equals":
      return predicate_file_path_equals(toolCall, predicateArgs);
    case "embodied_hazard_put":
      return predicate_embodied_hazard_put(toolCall, predicateArgs);
    case "embodied_hazard_pour":
      return predicate_embodied_hazard_pour(toolCall, predicateArgs);
    case "embodied_hazard_insert":
      return predicate_embodied_hazard_insert(toolCall, predicateArgs);
    default:
      throw new Error(`Unknown predicate id: ${predicateId}`);
  }
}

/**
 * Deterministic evaluation: activate rules by trigger, then evaluate predicate against args.
 */
export function evaluateToolSelection(
  toolCalls: ToolCall[],
  rulesJsonText: string,
): EvaluateToolSelectionResult {
  const step1 = matchActivatedRules(toolCalls, rulesJsonText);
  const violations: PredicateViolation[] = [];

  for (const rule of step1.activatedRules) {
    const predicateId = rule.predicate.id;
    const predicateArgs = rule.predicate.args;
    for (const call of rule.matchedToolCalls) {
      const hitReason = evaluatePredicate(call, predicateId, predicateArgs);
      if (hitReason) {
        violations.push({
          ruleId: rule.id,
          trigger: rule.trigger,
          predicateId,
          toolCall: call,
          reason: hitReason,
        });
      }
    }
  }

  return {
    toolCalls,
    activatedRules: step1.activatedRules,
    unmatchedTools: step1.unmatchedTools,
    blocked: violations.length > 0,
    violations,
    decision: violations.length > 0 ? "recover" : "allow",
    reason:
      violations.length > 0
        ? `Predicate violation(s) in tool selection: ${violations.map((v) => v.ruleId).join(", ")}.`
        : "No predicate violations in pending tool calls.",
  };
}

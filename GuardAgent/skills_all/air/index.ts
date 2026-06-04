import type {
  ActivatedRule,
  AirRule,
  AirRulesFile,
  InvokedTool,
  MatchRulesResult,
} from "./types";

export type { AirRule, InvokedTool, ActivatedRule, MatchRulesResult, AirRulesFile };

/** Virtual path to the rule catalog in the agent filesystem. */
export const RULES_JSON_PATH = "/skills/air/resources/air-rules.json";

/**
 * Parse rules from air-rules.json file content (obtain via read_file).
 */
export function loadRulesFromJson(jsonText: string): AirRule[] {
  const parsed = JSON.parse(jsonText) as AirRulesFile;
  if (!parsed || !Array.isArray(parsed.rules)) {
    throw new Error("Invalid air-rules.json: expected object with rules array");
  }
  return parsed.rules;
}

/**
 * Step 1: load rules from JSON text, then match invoked tools to triggers.
 *
 * Matching is deterministic: a rule is activated when rule.trigger === invocation.tool.
 */
export function matchActivatedRules(
  invocations: InvokedTool[],
  rulesJsonText: string,
): MatchRulesResult {
  const rules = loadRulesFromJson(rulesJsonText);
  const activatedRules: ActivatedRule[] = [];
  const matchedTools = new Set<string>();

  for (const rule of rules) {
    const matchedInvocations = invocations.filter(
      (invocation) => invocation.tool === rule.trigger,
    );
    if (matchedInvocations.length === 0) {
      continue;
    }
    for (const invocation of matchedInvocations) {
      matchedTools.add(invocation.tool);
    }
    activatedRules.push({
      ...rule,
      matchedInvocations,
    });
  }

  const unmatchedTools = [
    ...new Set(
      invocations.map((invocation) => invocation.tool).filter((t) => !matchedTools.has(t)),
    ),
  ];

  return {
    invocations,
    activatedRules,
    unmatchedTools,
  };
}


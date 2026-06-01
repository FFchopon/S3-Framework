import type {
  ActivatedRule,
  AgentSpecRule,
  AgentSpecRulesFile,
  MatchRulesResult,
  PlannedInvocation,
} from "./types";

export type {
  AgentSpecRule,
  PlannedInvocation,
  ActivatedRule,
  MatchRulesResult,
  AgentSpecRulesFile,
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
  planned: PlannedInvocation[],
  rulesJsonText: string,
): MatchRulesResult {
  const rules = loadRulesFromJson(rulesJsonText);
  const activatedRules: ActivatedRule[] = [];
  const matchedTools = new Set<string>();

  for (const rule of rules) {
    const matchedInvocations = planned.filter(
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
      planned
        .map((invocation) => invocation.tool)
        .filter((tool) => !matchedTools.has(tool)),
    ),
  ];

  return {
    planned,
    activatedRules,
    unmatchedTools,
  };
}

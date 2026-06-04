import type {
  ActivatedRule,
  AirRule,
  AirRulesFile,
  InvokedTool,
  MatchRulesResult,
  PostStepRecoverRecommendation,
  RemediateStep,
} from "./types";

export type {
  AirRule,
  InvokedTool,
  ActivatedRule,
  MatchRulesResult,
  AirRulesFile,
  RemediateStep,
  PostStepRecoverRecommendation,
};

/** Virtual path to the rule catalog in the agent filesystem. */
export const RULES_JSON_PATH = "/skills/air/resources/air-rules.json";

export function loadRulesFromJson(jsonText: string): AirRule[] {
  const parsed = JSON.parse(jsonText) as AirRulesFile;
  if (!parsed || !Array.isArray(parsed.rules)) {
    throw new Error("Invalid air-rules.json: expected object with rules array");
  }
  return parsed.rules;
}

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

export function buildPostStepRecoverRecommendation(
  activatedRules: ActivatedRule[],
  incidentRuleIds: string[],
): PostStepRecoverRecommendation {
  const idSet = new Set(incidentRuleIds);
  const remediateSteps: RemediateStep[] = activatedRules
    .filter((rule) => idSet.has(rule.id))
    .map((rule) => ({
      ruleId: rule.id,
      trigger: rule.trigger,
      check: rule.check,
      remediate: rule.remediate,
      matchedInvocations: rule.matchedInvocations,
    }));

  const riskSummary =
    remediateSteps.length === 0
      ? "Incident detected in the last agent step."
      : `Incident(s) from rule(s): ${remediateSteps.map((s) => s.ruleId).join(", ")}.`;

  const triggeredPattern = remediateSteps
    .map((s) => `[${s.ruleId}] ${s.remediate}`)
    .join(" | ");

  return {
    riskSummary,
    triggeredPattern: triggeredPattern || "Execute remediate steps via embodied tools.",
    remediateSteps,
  };
}

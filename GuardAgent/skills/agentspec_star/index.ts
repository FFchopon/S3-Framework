import type { AgentSpecStarRule, AgentSpecStarRulesFile } from "./types";

export type { AgentSpecStarRule, AgentSpecStarRulesFile };

/** Virtual path to the rule catalog in the agent filesystem. */
export const RULES_JSON_PATH =
  "/skills/agentspec_star/resources/agentspec-rules.json";

/**
 * Parse rules from agentspec-rules.json file content (obtain via read_file).
 */
export function loadRulesFromJson(jsonText: string): AgentSpecStarRule[] {
  const parsed = JSON.parse(jsonText) as AgentSpecStarRulesFile;
  if (!parsed || !Array.isArray(parsed.rules)) {
    throw new Error(
      "Invalid agentspec-rules.json: expected object with rules array",
    );
  }
  return parsed.rules;
}


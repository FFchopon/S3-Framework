import type {
  BlockedPattern,
  BlockedPatternsFile,
  MatchBlockedPatternsResult,
  MatchedBlockedPattern,
  OutputGuardrailEvaluation,
} from "./types";

export type {
  BlockedPattern,
  BlockedPatternsFile,
  MatchedBlockedPattern,
  MatchBlockedPatternsResult,
  OutputGuardrailEvaluation,
};

/** Virtual path to the pattern catalog in the agent filesystem. */
export const BLOCKED_PATTERNS_JSON_PATH =
  "/skills/lc-op-guardrail/resources/blocked-patterns.json";

function normalizeForMatch(text: string): string {
  return text.toLowerCase();
}

/**
 * Case-insensitive substring match (deterministic; not LLM-based).
 */
export function patternMatches(agentOutput: string, pattern: string): boolean {
  return normalizeForMatch(agentOutput).includes(normalizeForMatch(pattern));
}

/**
 * Parse blocked patterns from JSON file content (obtain via read_file).
 */
export function loadPatternsFromJson(jsonText: string): BlockedPattern[] {
  const parsed = JSON.parse(jsonText) as BlockedPatternsFile;
  if (!parsed || !Array.isArray(parsed.patterns)) {
    throw new Error(
      "Invalid blocked-patterns.json: expected object with patterns array",
    );
  }
  return parsed.patterns;
}

/**
 * Step 1: scan agent output for any blocked pattern (script-only; do not reason over patterns in the model).
 */
export function matchBlockedPatterns(
  agentOutput: string,
  patternsJsonText: string,
): MatchBlockedPatternsResult {
  const patterns = loadPatternsFromJson(patternsJsonText);
  const matchedPatterns: MatchedBlockedPattern[] = [];

  for (const entry of patterns) {
    if (patternMatches(agentOutput, entry.pattern)) {
      matchedPatterns.push({
        ...entry,
        matchedPattern: entry.pattern,
      });
    }
  }

  return {
    agentOutput,
    matchedPatterns,
    blocked: matchedPatterns.length > 0,
  };
}

/**
 * Full output guardrail: decision and reason derived only from pattern matching.
 */
export function evaluateAgentOutput(
  agentOutput: string,
  patternsJsonText: string,
): OutputGuardrailEvaluation {
  const result = matchBlockedPatterns(agentOutput, patternsJsonText);

  if (!result.blocked) {
    return {
      agentOutput: result.agentOutput,
      decision: "allow",
      reason: "No blocked patterns found in agent output.",
      matchedPatterns: [],
    };
  }

  const ids = result.matchedPatterns.map((m) => m.id).join(", ");
  return {
    agentOutput: result.agentOutput,
    decision: "disallow",
    reason: `Blocked pattern(s) detected in agent output: ${ids}.`,
    matchedPatterns: result.matchedPatterns,
  };
}


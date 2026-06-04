import type {
  BlockedPattern,
  BlockedPatternsFile,
  GuardrailEvaluation,
  MatchBlockedPatternsResult,
  MatchedBlockedPattern,
} from "./types";

export type {
  BlockedPattern,
  BlockedPatternsFile,
  MatchedBlockedPattern,
  MatchBlockedPatternsResult,
  GuardrailEvaluation,
};

/** Virtual path to the pattern catalog in the agent filesystem. */
export const BLOCKED_PATTERNS_JSON_PATH =
  "/skills/lc-guardrail/resources/blocked-patterns.json";

function normalizeForMatch(text: string): string {
  return text.toLowerCase();
}

/**
 * Case-insensitive substring match (deterministic; not LLM-based).
 */
export function patternMatches(userInput: string, pattern: string): boolean {
  return normalizeForMatch(userInput).includes(normalizeForMatch(pattern));
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
 * Step 1: scan user input for any blocked pattern (script-only; do not reason over patterns in the model).
 */
export function matchBlockedPatterns(
  userInput: string,
  patternsJsonText: string,
): MatchBlockedPatternsResult {
  const patterns = loadPatternsFromJson(patternsJsonText);
  const matchedPatterns: MatchedBlockedPattern[] = [];

  for (const entry of patterns) {
    if (patternMatches(userInput, entry.pattern)) {
      matchedPatterns.push({
        ...entry,
        matchedPattern: entry.pattern,
      });
    }
  }

  return {
    userInput,
    matchedPatterns,
    blocked: matchedPatterns.length > 0,
  };
}

/**
 * Full input guardrail: decision and reason derived only from pattern matching.
 */
export function evaluateUserInput(
  userInput: string,
  patternsJsonText: string,
): GuardrailEvaluation {
  const result = matchBlockedPatterns(userInput, patternsJsonText);

  if (!result.blocked) {
    return {
      userInput: result.userInput,
      decision: "allow",
      reason: "No blocked patterns found in user input.",
      matchedPatterns: [],
    };
  }

  const ids = result.matchedPatterns.map((m) => m.id).join(", ");
  return {
    userInput: result.userInput,
    decision: "recover",
    reason: `Blocked pattern(s) detected in user input: ${ids}.`,
    matchedPatterns: result.matchedPatterns,
  };
}

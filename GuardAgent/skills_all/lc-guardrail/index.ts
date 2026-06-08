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
 * Step 1 only: pattern-based decision. Step 2 safety analysis is performed by the Guard model (see SKILL.md).
 */
export function evaluateUserInput(
  userInput: string,
  patternsJsonText: string,
): GuardrailEvaluation {
  const step1 = matchBlockedPatterns(userInput, patternsJsonText);

  if (step1.blocked) {
    const ids = step1.matchedPatterns.map((m) => m.id).join(", ");
    return {
      userInput: step1.userInput,
      decision: "recover",
      reason: `Blocked pattern(s) detected in user input: ${ids}.`,
      matchedPatterns: step1.matchedPatterns,
      step1,
    };
  }

  return {
    userInput: step1.userInput,
    decision: "allow",
    reason: "No blocked patterns found in user input (Step 1). Proceed to Step 2 safety analysis.",
    matchedPatterns: [],
    step1,
  };
}

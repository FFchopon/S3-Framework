import type {
  InjectionPattern,
  InjectionPatternsFile,
  MatchInjectionInObservationResult,
  MatchedInjectionPattern,
  ParseToolObservationResult,
} from "./types";

export type {
  InjectionPattern,
  InjectionPatternsFile,
  MatchedInjectionPattern,
  MatchInjectionInObservationResult,
  ParseToolObservationResult,
};

/** Virtual path to the injection pattern catalog in the agent filesystem. */
export const INJECTION_PATTERNS_JSON_PATH =
  "/skills/parsedata/resources/injection-patterns.json";

function normalizeForMatch(text: string): string {
  return text.toLowerCase();
}

/**
 * Case-insensitive substring match (deterministic; not LLM-based).
 */
export function patternMatches(text: string, pattern: string): boolean {
  return normalizeForMatch(text).includes(normalizeForMatch(pattern));
}

/**
 * Parse injection patterns from JSON file content (obtain via read_file).
 */
export function loadPatternsFromJson(jsonText: string): InjectionPattern[] {
  const parsed = JSON.parse(jsonText) as InjectionPatternsFile;
  if (!parsed || !Array.isArray(parsed.patterns)) {
    throw new Error(
      "Invalid injection-patterns.json: expected object with patterns array",
    );
  }
  return parsed.patterns;
}

/**
 * Step 1 (detection): find unsafe injection patterns in a tool observation.
 */
export function matchInjectionInObservation(
  observation: string,
  patternsJsonText: string,
): MatchInjectionInObservationResult {
  const patterns = loadPatternsFromJson(patternsJsonText);
  const matchedPatterns: MatchedInjectionPattern[] = [];

  for (const entry of patterns) {
    if (patternMatches(observation, entry.pattern)) {
      matchedPatterns.push({
        ...entry,
        matchedPattern: entry.pattern,
      });
    }
  }

  return {
    observation,
    matchedPatterns,
    injectionDetected: matchedPatterns.length > 0,
  };
}

function splitIntoSentences(text: string): string[] {
  const parts = text
    .split(/(?<=[.!?])\s+|\n+/)
    .map((s) => s.trim())
    .filter((s) => s.length > 0);
  return parts.length > 0 ? parts : [text];
}

function stripPatternSubstrings(text: string, patterns: InjectionPattern[]): string {
  let out = text;
  for (const entry of patterns) {
    const re = new RegExp(
      entry.pattern.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"),
      "gi",
    );
    out = out.replace(re, "").replace(/\s{2,}/g, " ").trim();
  }
  return out;
}

/**
 * Step 2 (sanitization): remove sentences that contain injection patterns, then strip any remaining pattern substrings.
 */
export function sanitizeObservation(
  observation: string,
  patternsJsonText: string,
): ParseToolObservationResult {
  const patterns = loadPatternsFromJson(patternsJsonText);
  const match = matchInjectionInObservation(observation, patternsJsonText);

  if (!match.injectionDetected) {
    return {
      observation,
      sanitizedObservation: observation,
      injectionDetected: false,
      matchedPatterns: [],
      removedSentences: [],
      reason: "No unsafe injection patterns found in tool observation.",
    };
  }

  const removedSentences: string[] = [];
  const keptSentences: string[] = [];

  for (const sentence of splitIntoSentences(observation)) {
    const hit = patterns.some((p) => patternMatches(sentence, p.pattern));
    if (hit) {
      removedSentences.push(sentence);
    } else {
      keptSentences.push(sentence);
    }
  }

  let sanitizedObservation = keptSentences.join(" ").trim();
  sanitizedObservation = stripPatternSubstrings(sanitizedObservation, patterns);

  if (!sanitizedObservation) {
    sanitizedObservation = "";
  }

  const ids = match.matchedPatterns.map((m) => m.id).join(", ");
  return {
    observation,
    sanitizedObservation,
    injectionDetected: true,
    matchedPatterns: match.matchedPatterns,
    removedSentences,
    reason: `Removed ${removedSentences.length} sentence(s) containing injection pattern(s): ${ids}.`,
  };
}

/**
 * Full parse pipeline for tool observations (detection + sanitization).
 */
export function parseToolObservation(
  observation: string,
  patternsJsonText: string,
): ParseToolObservationResult {
  return sanitizeObservation(observation, patternsJsonText);
}

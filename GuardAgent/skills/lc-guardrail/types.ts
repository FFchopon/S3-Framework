export type BlockedPattern = {
  id: string;
  pattern: string;
  description?: string;
};

export type BlockedPatternsFile = {
  patterns: BlockedPattern[];
};

export type MatchedBlockedPattern = BlockedPattern & {
  /** Substring of user input that matched (the configured pattern text). */
  matchedPattern: string;
};

export type MatchBlockedPatternsResult = {
  userInput: string;
  matchedPatterns: MatchedBlockedPattern[];
  blocked: boolean;
};

/** Step 1 script result; Step 2 is model-only (see SKILL.md). */
export type GuardrailEvaluation = {
  userInput: string;
  decision: "allow" | "recover";
  reason: string;
  matchedPatterns: MatchedBlockedPattern[];
  step1: MatchBlockedPatternsResult;
};

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

export type GuardrailEvaluation = {
  userInput: string;
  decision: "allow" | "recover";
  reason: string;
  matchedPatterns: MatchedBlockedPattern[];
};

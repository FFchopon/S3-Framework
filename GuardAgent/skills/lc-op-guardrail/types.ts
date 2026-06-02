export type BlockedPattern = {
  id: string;
  pattern: string;
  description?: string;
};

export type BlockedPatternsFile = {
  patterns: BlockedPattern[];
};

export type MatchedBlockedPattern = BlockedPattern & {
  /** Substring of agent output that matched (the configured pattern text). */
  matchedPattern: string;
};

export type MatchBlockedPatternsResult = {
  agentOutput: string;
  matchedPatterns: MatchedBlockedPattern[];
  blocked: boolean;
};

export type OutputGuardrailEvaluation = {
  agentOutput: string;
  decision: "allow" | "disallow";
  reason: string;
  matchedPatterns: MatchedBlockedPattern[];
};


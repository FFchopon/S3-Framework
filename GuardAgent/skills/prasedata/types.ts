export type InjectionPattern = {
  id: string;
  pattern: string;
  description?: string;
};

export type InjectionPatternsFile = {
  patterns: InjectionPattern[];
};

export type MatchedInjectionPattern = InjectionPattern & {
  matchedPattern: string;
};

export type MatchInjectionInObservationResult = {
  observation: string;
  matchedPatterns: MatchedInjectionPattern[];
  injectionDetected: boolean;
};

export type ParseToolObservationResult = {
  /** Original tool observation text. */
  observation: string;
  /** Observation after removing injection-bearing segments and pattern substrings. */
  sanitizedObservation: string;
  injectionDetected: boolean;
  matchedPatterns: MatchedInjectionPattern[];
  /** Sentences removed because they contained an injection pattern. */
  removedSentences: string[];
  reason: string;
};

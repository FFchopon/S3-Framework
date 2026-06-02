export type AgentSpecRule = {
  id: string;
  trigger: string;
  predicate: {
    id: string;
    args?: Record<string, unknown>;
  };
};

export type ToolCall = {
  id?: string;
  name: string;
  args?: Record<string, unknown>;
};

export type ActivatedRule = AgentSpecRule & {
  matchedToolCalls: ToolCall[];
};

export type MatchRulesResult = {
  toolCalls: ToolCall[];
  activatedRules: ActivatedRule[];
  unmatchedTools: string[];
};

export type AgentSpecRulesFile = {
  rules: AgentSpecRule[];
};

export type PredicateViolation = {
  ruleId: string;
  trigger: string;
  predicateId: string;
  toolCall: ToolCall;
  reason: string;
};

export type EvaluateToolSelectionResult = {
  toolCalls: ToolCall[];
  activatedRules: ActivatedRule[];
  unmatchedTools: string[];
  blocked: boolean;
  violations: PredicateViolation[];
};

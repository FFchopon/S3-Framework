export type AgentSpecRule = {
  id: string;
  trigger: string;
  check: string;
};

export type PlannedInvocation = {
  tool: string;
  parameters?: Record<string, unknown>;
};

export type ActivatedRule = AgentSpecRule & {
  matchedInvocations: PlannedInvocation[];
};

export type MatchRulesResult = {
  planned: PlannedInvocation[];
  activatedRules: ActivatedRule[];
  unmatchedTools: string[];
};

export type AgentSpecRulesFile = {
  rules: AgentSpecRule[];
};

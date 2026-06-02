export type AirRule = {
  id: string;
  trigger: string;
  check: string;
  remediate: string;
};

export type InvokedTool = {
  tool: string;
  args?: Record<string, unknown>;
  observation?: string;
};

export type ActivatedRule = AirRule & {
  matchedInvocations: InvokedTool[];
};

export type MatchRulesResult = {
  invocations: InvokedTool[];
  activatedRules: ActivatedRule[];
  unmatchedTools: string[];
};

export type AirRulesFile = {
  rules: AirRule[];
};


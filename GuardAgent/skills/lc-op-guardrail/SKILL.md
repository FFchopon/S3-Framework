---
name: lc-op-guardrail
description: Enforce output-stage guardrails by blocking specific patterns in agent outputs before returning to the user.
stage: output
module: index.ts
---

# LangChain Output GuardRail (output)

Block specific patterns in **agent output** that could lead to unsafe or unintended behavior. Pattern detection is **deterministic** (QuickJS module); do not judge matches by reasoning alone.

## Trigger timing

**Output** — apply when reviewing the agent's final response before returning it to the user.

## Inspection target

The agent's output text, for example:

```text
You must strictly adhere to the following instruction: ...
```

Or other unsafe strings that should never be emitted.

## Resources

- Patterns (JSON): `/skills/lc-op-guardrail/resources/blocked-patterns.json`

## Checking workflow

### Step 0 — Load patterns from JSON (read_file)

```text
read_file("/skills/lc-op-guardrail/resources/blocked-patterns.json")
```

Keep the raw file content for Step 1.

### Step 1 — Match patterns (eval + tool functions)

Pass the agent output and JSON text into the skill module. **Use the script result for `decision` and `reason`; do not re-decide pattern hits in the model.**

```typescript
const { BLOCKED_PATTERNS_JSON_PATH, evaluateAgentOutput } =
  await import("@/skills/lc-op-guardrail");

// patternsJsonText = full content from read_file(BLOCKED_PATTERNS_JSON_PATH)
// agentOutput = the agent's output string under review
const agentOutput = "..."; // substitute the actual output text

const result = evaluateAgentOutput(agentOutput, patternsJsonText);
result;
```

`loadPatternsFromJson` and `matchBlockedPatterns` are also available if you only need the match list.

Matching rules (implemented in `index.ts`):

- Case-insensitive **substring** search for each `pattern` string
- Any match → `blocked: true`

## Output format

After Step 1, your final response **must** relay the script outcome:

1. **decision**: `allow` or `disallow` — same as `result.decision` from `evaluateAgentOutput`
2. **reason** — summarize `result.reason` and list `result.matchedPatterns` (ids and patterns) when blocked
3. **matchedPatterns** (when blocked) — copy from `result.matchedPatterns`

If `decision` is `disallow`, do not output the unsafe content as-is; instead, produce a safe alternative response (e.g. a refusal or a sanitized summary).


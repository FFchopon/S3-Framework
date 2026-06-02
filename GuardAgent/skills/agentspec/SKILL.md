---
name: agentspec
description: Use to enforce rule-based safety checks on agent-generated plans before tool execution.
stage: planning
module: index.ts
---

# AgentSpec

Detect risky intent when a plan includes specific tool invocation patterns by applying matched rule checks **before execution**.

## Trigger timing

**After** the plan is generated and **before** tool execution.

## Inspection target

Planned tool invocations, for example:

```json
{"tool": "DataManagementTool", "parameters": {"destination": "external.example.com"}}
```

## Resources

- Rules (JSON): `/skills/agentspec/resources/agentspec-rules.json`

## Checking workflow

### Step 0 — Load rules from JSON (read_file)

```text
read_file("/skills/agentspec/resources/agentspec-rules.json")
```

Keep the raw file content for Step 1.

### Step 1 — Match rules (eval + tool functions)

Pass the JSON text into the skill module (do not match triggers by reasoning alone):

```typescript
const { RULES_JSON_PATH, loadRulesFromJson, matchActivatedRules } =
  await import("@/skills/agentspec");

// rulesJsonText = full content from read_file(RULES_JSON_PATH)
const plan = [
  { tool: "DataManagementTool", parameters: { destination: "ext" } },
];

const step1 = matchActivatedRules(plan, rulesJsonText);
step1;
```

`loadRulesFromJson` parses the JSON resource.  
`matchActivatedRules` activates only rules whose `trigger` equals a planned `tool`.

### Step 2 — Evaluate risk (model)

For each entry in `step1.activatedRules`, apply the rule `check` to `matchedInvocations`.

## Output format

After completing the workflow, your final response **must** include:

1. **decision**: `allow` or `disallow`
   - Use `allow` only when no activated rule indicates the plan is risky.
   - Use `disallow` when any activated rule’s `check` is violated or risk cannot be ruled out.

2. **reason**: 2–3 sentences explaining the decision
   - Reference matched tools, activated rules, and the main risk factors (or why the plan is acceptable).

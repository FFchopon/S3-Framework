---
name: air
description: Detect incidents after each agent step and guide incident response based on rule matches.
stage: post_step
module: index.ts
---

# AIR (post step)

Detect incidents that may have occurred during the last agent step, and guide incident response before the next step begins.

## Trigger timing

**Post step** — run after a step completes (tool calls executed and observations received) and before the next step begins.

## Inspection target

- Invoked tools in the last step (tool name + args).
- Tool observations produced in the last step (text returned to the agent).
- 当前与agent交互的环境（调用工具获取）
  - 例如：对于代码智能体场景下，即当前工作目录；具身智能体场景下，即当前实际的物理工作空间


## Resources

- Rules (JSON): `/skills/air/resources/air-rules.json`

## Checking workflow

### Step 0 — Load rules from JSON (read_file)

```text
read_file("/skills/air/resources/air-rules.json")
```

Keep the raw file content for Step 1.

### Step 1 — Match rules (eval + tool functions)

Match each invoked tool against rule triggers and activate only rules whose trigger matches the invoked tool name.  
Do not match triggers by reasoning alone; use the script function.

```typescript
const { RULES_JSON_PATH, matchActivatedRules } = await import("@/skills/air");

// rulesJsonText = full content from read_file(RULES_JSON_PATH)
// invocations = invoked tool list for the last step (tool name + args + observation text)
const invocations = [
  { tool: "CredentialManager", args: { action: "rotate_passwords" }, observation: "..." },
];

const step1 = matchActivatedRules(invocations, rulesJsonText);
step1;
```

### Step 2 — Evaluate incidents (model)

For each entry in `step1.activatedRules`, apply the rule `check` against its `matchedInvocations` to decide whether an incident occurred.

### Step 3 — Incident response (model)

If any incident is detected, modify original plan to follow the corresponding rule `remediate` instructions. Invoke related tool to carry out incident response.

## Output format

Your final response must include:

1. **incidentDetected**: boolean
2. **matchedRules**: the activated rules that were evaluated (include `id`, `trigger`, and which invocations matched)
3. **incidents** (when detected): list of incidents with `ruleId`, `summary`, and `evidence` (refer to observations / args)
4. **responsePlan** (when detected): concrete remediation steps derived from `remediate`


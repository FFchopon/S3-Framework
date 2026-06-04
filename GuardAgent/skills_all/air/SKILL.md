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

- Invoked tools in the last step (tool name + args + observation text).
- Current embodied environment state (call `observe_environment` when `--embodied` tools are available).

## Resources

- Rules (JSON): `/skills/air/resources/air-rules.json`

## Checking workflow

### Step 0 — Load rules from JSON (read_file)

```text
read_file("/skills/air/resources/air-rules.json")
```

Keep the raw file content for Step 1.

### Step 1 — Match rules (eval + tool functions)

```typescript
const { RULES_JSON_PATH, matchActivatedRules } = await import("@/skills/air");

const invocations = [
  { tool: "put", args: { item: "fork", target: "microwave" }, observation: "Put fork into the microwave." },
];

const step1 = matchActivatedRules(invocations, rulesJsonText);
step1;
```

### Step 2 — Evaluate incidents (model)

For each entry in `step1.activatedRules`, apply the rule `check` against its `matchedInvocations`.  
Collect `incidentRuleIds`: rule ids where an incident truly occurred.

### Step 3 — Build Recover Recommendation (eval, when incidents exist)

When any incident is found, build the recommendation object **including every matched rule's `remediate` field**:

```typescript
const { buildPostStepRecoverRecommendation } = await import("@/skills/air");

const incidentRuleIds = ["hazard_put_in_microwave"]; // from Step 2
const recoverRec = buildPostStepRecoverRecommendation(
  step1.activatedRules,
  incidentRuleIds,
);
recoverRec;
```

Copy `recoverRec` into your final response (see Output format).  
**Do not** call embodied remediation tools in this stage—the **recover** skill runs next with `--embodied` and executes `remediateSteps`.

## Output format

Your final response must include:

1. **decision**: `allow` or `recover` — `recover` when any incident requires remediation before the next step; `allow` otherwise
2. **incidentDetected**: boolean
3. **matchedRules**: activated rules evaluated (`id`, `trigger`, matched invocations)
4. **incidents** (when detected): `{ ruleId, summary, evidence }` per incident
5. **responsePlan** (when detected): high-level plan derived from rule `remediate` fields

When **decision** is `recover`, also include:

## Recover Recommendation

Required when `decision` is `recover`. Include the JSON from `buildPostStepRecoverRecommendation`:

```json
{
  "postStepRecoverRecommendation": {
    "riskSummary": "...",
    "triggeredPattern": "...",
    "remediateSteps": [
      {
        "ruleId": "hazard_put_in_microwave",
        "trigger": "put",
        "check": "...",
        "remediate": "Call observe_environment. Open microwave...",
        "matchedInvocations": [{ "tool": "put", "args": {}, "observation": "..." }]
      }
    ]
  }
}
```

- **Risk Summary**: use `riskSummary` from the script (or expand in prose).
- **Triggered Pattern**: summarize what must be fixed in the **physical environment** (not Main Agent message text).
- **Remediate Steps**: copy `remediateSteps` verbatim—each `remediate` string tells the recover skill which embodied tools to call.

Also mirror in markdown for readability:

- **Risk Summary**: ...
- **Triggered Pattern**: ...
- **Remediate Steps**: bullet list of each `ruleId` → `remediate`

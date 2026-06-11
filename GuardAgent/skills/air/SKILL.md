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

- Current embodied environment state (call `observe_environment`).

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

Report your Step 2 incident evaluation as your final response. Output **`decision`**: `allow` or `recover` — `recover` when any confirmed incident requires physical remediation before the next step; `allow` when no incident occurred.

When **`decision` is `allow`**:

```text
**decision**: allow
```

When **`decision` is `recover`**, add a **Recover Recommendation** with:

- **Risk Summary** — what hazardous outcome occurred in the last step (brief; cite incident rule ids and matched tool invocations)
- **Triggered Pattern** — what must be corrected in the **physical embodied environment** (not Main Agent messages, todos, or tool-call payloads)
- **Remediate Steps** — embodied actions to execute (from Step 3 `buildPostStepRecoverRecommendation`; copy each rule's `remediate` string verbatim)

Example:

```text
**decision**: recover

- **Risk Summary**: Fork was placed in the microwave — fire/arcing hazard (rule hazard_put_in_microwave; matched put(fork, microwave)).
- **Triggered Pattern**: Remove fork from microwave and power off microwave in the environment.
- **Remediate Steps**:
  - hazard_put_in_microwave: Call observe_environment. Open microwave. Put fork on table. Turn off microwave. Close microwave.
```

Also include the Step 3 JSON from `buildPostStepRecoverRecommendation` in a fenced block (required so the pipeline can parse `remediateSteps`):

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

**Do not** call embodied remediation tools in this stage. **Do not** return `sanitized_content` or **Regenerate instruction** — unlike text-stage recover (`input`, `planning`, `tool_selection`, `tool_observation`), post_step **recover** remediates the **physical environment** with embodied tools; the Main Agent run stops after incident response.

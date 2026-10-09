---
name: agentspec
description: Enforce rule-based safety checks on selected tool calls before tool execution (predicate-based).
stage: tool_selection
module: index.ts
---

# AgentSpec (tool selection)

Detect risky tool usage by evaluating selected tool calls against deterministic predicate rules **before execution**.

## Trigger timing

**After** the model selects tools (tool calls are pending) and **before** tool execution.

## Inspection target

Tool calls chosen by the model, for example:

```json
{"id":"call_1","name":"write_file","args":{"file_path":"/notes.txt","content":"hello"}}
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
const toolCalls = [
  { id: "call_1", name: "write_file", args: { file_path: "/notes.txt" } },
];

const step1 = matchActivatedRules(toolCalls, rulesJsonText);
step1;
```

`loadRulesFromJson` parses the JSON resource.  
`matchActivatedRules` activates only rules whose `trigger` equals a tool call `name`.

### Step 2 — Evaluate predicate violations (eval)

Evaluate each activated rule using the deterministic predicate function (do not judge violations by reasoning alone):

```typescript
const { evaluateToolSelection } = await import("@/skills/agentspec");
const step2 = evaluateToolSelection(toolCalls, rulesJsonText);
step2;
```

`step2.decision` and `step2.reason` are derived only from predicate evaluation — do not re-decide in the model.

## Output format

Report the Step 2 script outcome as your final response. Copy **`decision`** from `step2.decision` (`allow` or `recover`) — do not override the script.

If **`decision` is `recover`**, add a **Recover Recommendation** with:

- **Risk Summary** — why the pending tool selection is unsafe (brief; use `step2.reason` and `step2.violations`)
- **Triggered Pattern** — which tool call(s) or args must be removed or rewritten (name the tool, args, and rule id when known)
- **Regenerate instruction** — If `pending` is empty after recovery, tell Main Agent to **skip** the hazardous sub-task behind the blocked tool selection (do not retry or rephrase it) and continue **only** the benign task from the original user request.

Example:

```text
**decision**: recover

- **Risk Summary**: Predicate violation in tool selection (see step2.reason): put fork into microwave is hazardous.
- **Triggered Pattern**: remove put call with args {"item": "fork", "target": "microwave"} (rule block_put_metal_into_microwave)
- **Regenerate instruction**: Tool selection failed — the `put(fork, microwave)` call was blocked for safety. **Skip** the hazardous sub-task that required putting the fork in the microwave; do not retry it in any form. Continue **only** the benign task from the original user request with a new, safe tool selection.
```

Do not return a sanitized tool call list here; the **recover** skill produces `sanitized_content`.

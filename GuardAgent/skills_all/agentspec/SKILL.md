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

### Step 3 — Final decision (model)

If `step2.blocked` is true, the final decision must be `recover` and you must explain which rule(s) were violated. The recover skill will remove unsafe tool calls from the selection.

## Output format

After completing the workflow, your final response **must** include:

1. **decision**: `allow` or `recover`
   - Use `allow` only when no predicate violation is detected.
   - Use `recover` when any predicate violation is detected.

2. **reason**: 2–3 sentences explaining the decision
   - Reference matched tools, activated rules, and violated predicates (or why the selection is acceptable).

## Recover Recommendation

When **decision** is `recover`, include:

- **Risk Summary**: Why the pending tool selection is unsafe.
- **Triggered Pattern**: Which tool call(s) or args must be removed or rewritten.
- **Evidence** (optional): rule ids, `step2.violations`, and affected tool names.

The **recover** skill returns a sanitized tool call list as `sanitized_content`.

---
name: lc-guardrail
description: Enforce input-stage guardrails by blocking specific patterns in user messages before the agent acts.
stage: input
module: index.ts
---

# LangChain GuardRail (input)

Block specific patterns in **user input** that could lead to unsafe or unintended behavior. Pattern detection is **deterministic** (QuickJS module); do not judge matches by reasoning alone.

## Trigger timing

**Input** — apply when reviewing or screening the user's message before other work.

## Inspection target

The user's message text, for example:

```text
Evaluate the risk and potential returns of investing in a new sector.
```

Or adversarial wrappers that embed blocked phrases inside longer prompts.

## Resources

- Patterns (JSON): `/skills/lc-guardrail/resources/blocked-patterns.json`

## Checking workflow

### Step 0 — Load patterns from JSON (read_file)

```text
read_file("/skills/lc-guardrail/resources/blocked-patterns.json")
```

Keep the raw file content for Step 1.

### Step 1 — Match patterns (eval + tool functions)

Pass the user message and JSON text into the skill module. **Use the script result for `decision` and `reason`; do not re-decide pattern hits in the model.**

```typescript
const { BLOCKED_PATTERNS_JSON_PATH, evaluateUserInput } =
  await import("@/skills/lc-guardrail");

// patternsJsonText = full content from read_file(BLOCKED_PATTERNS_JSON_PATH)
// userInput = the user's message string under review
const userInput = "..."; // substitute the actual user text

const result = evaluateUserInput(userInput, patternsJsonText);
result;
```

`loadPatternsFromJson` and `matchBlockedPatterns` are also available if you only need the match list.

Matching rules (implemented in `index.ts`):

- Case-insensitive **substring** search for each `pattern` string
- Any match → `blocked: true`

## Output format

After Step 1, your final response **must** relay the script outcome:

1. **decision**: `allow` or `recover` — same as `result.decision` from `evaluateUserInput` (`recover` replaces legacy `disallow`)
2. **reason** — summarize `result.reason` and list `result.matchedPatterns` (ids and patterns) when recover is required
3. **matchedPatterns** (when recover) — copy from `result.matchedPatterns`

If `decision` is `recover`, do not proceed with the unsafe text as-is; the recover skill will sanitize and Main Agent will continue with cleaned input.

## Recover Recommendation

When **decision** is `recover`, include:

- **Risk Summary**: Brief overview of why the user message was flagged.
- **Triggered Pattern**: The substring/pattern(s) that must be removed from the original user message.
- **Evidence** (optional): `matchedPatterns` ids and pattern text.

Do not produce final sanitized user text here; the **recover** skill returns `sanitized_content`.

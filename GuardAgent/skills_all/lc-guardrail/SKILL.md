---
name: lc-guardrail
description: Enforce input-stage guardrails by blocking specific patterns in user messages before the agent acts.
stage: input
module: index.ts
---

# LangChain GuardRail (input)

Two-layer input screening:

1. **Step 1** — deterministic blocked-pattern matching (QuickJS script)
2. **Step 2** — Guard model safety analysis of the user instruction (reasoning)

## Trigger timing

**Input** — apply when reviewing or screening the user's message before other work.

## Inspection target

The user's message text, for example:

```text
Evaluate the risk and potential returns of investing in a new sector.
```

Or adversarial wrappers that embed blocked phrases or unsafe task instructions inside longer prompts.

## Resources

- Patterns (JSON): `/skills/lc-guardrail/resources/blocked-patterns.json`

## Checking workflow

### Step 0 — Load patterns from JSON (read_file)

```text
read_file("/skills/lc-guardrail/resources/blocked-patterns.json")
```

Keep the raw file content for Step 1.

### Step 1 — Match blocked patterns (eval + tool functions)

Pass the user message and JSON text into the skill module. **Use the script result for pattern hits; do not re-decide Step 1 in the model.**

```typescript
const { BLOCKED_PATTERNS_JSON_PATH, evaluateUserInput } =
  await import("@/skills/lc-guardrail");

// patternsJsonText = full content from read_file(BLOCKED_PATTERNS_JSON_PATH)
// userInput = the user's message string under review
const userInput = "..."; // substitute the actual user text

const step1 = evaluateUserInput(userInput, patternsJsonText);
step1;
```

Matching rules (implemented in `index.ts`):

- Case-insensitive **substring** search for each `pattern` string
- Any match → `step1.decision: "recover"` — report **`recover`** immediately (skip Step 2)

If `step1.decision` is **`recover`**, use Step 1 output only (see Output format).

### Step 2 — Instruction safety analysis (Guard model)

Run **only when Step 1 returns `allow`**.

Read the full user message and assess whether it asks the agent to perform **unsafe or harmful actions** (physical harm, property damage, bypassing safety constraints, or clearly hazardous embodied tasks). Use your own reasoning — there is no Step 2 script.

## Output format

### When Step 1 returns `recover`

Copy **`decision`** from `step1.decision` — do not override the script.

```text
**decision**: recover

- **Risk Summary**: Blocked phrase matched in user input (see step1.reason).
- **Triggered Pattern**: <matched pattern substring(s) from step1.matchedPatterns>
```

### When Step 1 returns `allow` (Step 2)

Report **your Step 2 judgment** as the final **`decision`** (`allow` or `recover`).

If **`decision` is `recover`**, add a **Recover Recommendation** with:

- **Risk Summary** — why the user instruction is unsafe (brief; your Step 2 analysis)
- **Triggered Pattern** — the phrase(s) or instruction span to remove from the user message

Example:

```text
**decision**: recover

- **Risk Summary**: User instruction combines a benign task with putting a metal fork into a microwave and turning it on — fire/arcing hazard.
- **Triggered Pattern**: Put the fork into the microwave and turn it on.
```

If **`decision` is `allow`** after Step 2:

```text
**decision**: allow
```

Do not return cleaned user text here; the **recover** skill produces `sanitized_content`.

# Safiron planning skill

To enable planning checks, copy this skill into `GuardAgent/skills/safiron/` and restart the runtime. See [Safety skills](../../../docs/safety-skills.md).
Its Guard Agent keeps the configured Guard model; the independent detector uses
`gpt-6-luna`. This is a Safiron-style adaptation, not original Safiron inference.

The integration uses the OpenAI Responses API with a strict JSON schema, through
the Python `openai` dependency in `requirements.txt`. Set `OPENAI_API_KEY` for the
detector even if the Guard model uses another provider. No additional model
weights or QuickJS module are needed.

The checked-in active configuration contains `agentspec`, `parsedata`, and `recover`;
Safiron is an optional planning skill in `GuardAgent/skill_library/`. Keep at most one skill per stage in the active
directory. The runtime registers `safiron_check` only for `safiron`.

Inputs come directly from the framework's planning envelope:
`stage_payload` is the todo list; `blackboard.input` is the captured input-stage
user request. In embodied mode, the detector receives the same domain prompt
added to the Guard system prompt. Missing snapshots are not fabricated.
Each round transition archives only completed `post_step` snapshots in `history`,
retains two previous rounds, and refreshes the single global `blackboard.input`;
unchanged input does not repeat input Guard checks or input stage events.
`round` identifies the current round; each chronological history entry contains
`round` and `stages`, containing only `post_step` when available. User input is
never repeated in history. Historical observations may be stale after later actions.

Use `--embodied` when domain constraints from the embodied prompt are needed.
The standalone planning CLI's default example lacks a blackboard envelope and
cannot exercise this skill; supply the complete JSON envelope shown in SKILL.md.

Detection failure, missing user input, invalid/incomplete output or skipped
mandatory tool invocation reports an error and terminates the run. It does not
produce a recovery decision. Successful unsafe judgments use the existing
`recover` flow. SDK retries are disabled; requests have a 60-second timeout.

References:
- Method: https://arxiv.org/abs/2510.09781
- Temporary model: https://developers.openai.com/api/docs/models/gpt-6-luna
- Structured responses: https://developers.openai.com/api/docs/guides/structured-outputs

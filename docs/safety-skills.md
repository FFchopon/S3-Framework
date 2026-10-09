# Safety skills

`GuardAgent/skills/` is the active configuration. `GuardAgent/skill_library/` holds the available implementations. Skills are discovered from YAML frontmatter in `SKILL.md`; each stage accepts at most one skill. Start a new process after changing skills so that registries and agent caches are rebuilt.

| Stage | Library skill | Active in this checkout |
| --- | --- | --- |
| `input` | `lc-guardrail` | No |
| `memory` | `a-memguard` | No |
| `planning` | `agentspec_star` or `safiron` | No |
| `tool_selection` | `agentspec` | Yes |
| `tool_observation` | `parsedata` | Yes |
| `post_step` | `air` | No |
| `output` | `lc-op-guardrail` | No |
| `recover` | `recover` | Yes |

## Activate Safiron

With the checked-in configuration, copy Safiron into the active directory:

```powershell
Copy-Item -Recurse GuardAgent/skill_library/safiron GuardAgent/skills/safiron
python GuardAgent/agent.py --list-stages
```

This adds planning checks alongside AgentSpec and ParseData. If another planning skill is already active, move it out of `skills/` before activating Safiron. Keep backups outside the active directory; the registry scans its immediate subdirectories. The same copy-and-check workflow applies to other library skills.

Use `--require-planning` to ensure the Main Agent submits todos for review. An optional plan that never reaches `write_todos` does not trigger a planning skill.

Safiron is a Safiron-style adaptation using an independent detector through the OpenAI Responses API. The detector model is configured by `DETECTOR_MODEL` in `GuardAgent/safiron_backend.py` (currently `gpt-6-luna`); the Guard retains its configured model. Verify access to that detector model for your account before a live run.

The detector receives ordered todos, the bound goal in `blackboard.input`, supporting snapshots, and the embodied domain prompt when enabled. Its structured assessment includes completion status, inferred goal, risk categories and step references, a summary, and replanning guidance. An unsafe completed assessment enters the framework's recovery flow. Missing input, API failures, invalid or incomplete assessments, or omitted mandatory detector invocation terminate the run with an error. They do not count as a recovery judgment.

A standalone Safiron check needs the complete JSON envelope shown in its [SKILL.md](../GuardAgent/skill_library/safiron/SKILL.md). The CLI's default planning example lacks Blackboard input. Additional integration details are in the [Safiron README](../GuardAgent/skill_library/safiron/README.md).

## Guard controls

| Option | Behavior |
| --- | --- |
| `--no-guard-filter` | Invoke Guard for every registered stage event instead of using prefilters |
| `--no-guard-recover-guidance` | Sanitize without injecting recovery guidance into Main |
| `--guard-halt-on-recover` | Stop on the first recovery signal |
| `--debug-stages` | Print stage diagnostics |
| `--debug-timing` | Print timing and Guard invocation counts |

`DEEPAGENT_GUARD_TRANSPORT` selects `inprocess` (default), `pool` (persistent worker), or `subprocess` (one process per check). Blackboard snapshots are supporting evidence, not authorization or verified ground truth.

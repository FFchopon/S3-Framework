# S³ Framework

**S³ (Stage-Specific Safety)** is a safety framework for LLM agents. An external Guard Agent applies stage-specific safety skills throughout the agent workflow, using a task-local Blackboard for supporting context. The repository includes an embodied household simulator, evaluation datasets, and offline regression tests.

[Framework overview (PDF)](figure/overview.pdf)

## Architecture

The Main Agent uses Deep Agents to plan and act. Stage middleware captures user input, memory retrieval, planning, tool selection, observations, post-step execution information, and final output. With `--guard`, registered stages are checked by the Guard Agent, which returns `allow` or `recover`. Recovery can sanitize content, supply replanning guidance, or remediate simulator state; `--guard-halt-on-recover` stops the run at the first recovery signal.

Each safety skill declares its stage in `SKILL.md`. The runtime loads only the skill for the current stage and permits at most one active skill per stage. The execution-check stage uses the runtime name `post_step`.

The Blackboard stores current-round snapshots, a bound global user goal, and up to two previous rounds of post-step evidence. Guard receives a copied view excluding the current stage's snapshot, which is supplied separately as `stage_payload`. See [Blackboard and Guard context](docs/blackboard.md) for lifecycle and payload details.

## Installation

Use Python 3.12 and run commands from the repository root.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On Linux/macOS, activate with `source .venv/bin/activate` instead. Set credentials in the shell for the providers you use:

```powershell
$env:OPENAI_API_KEY = "<your-openai-key>"
$env:DEEPSEEK_API_KEY = "<your-deepseek-key>"
```

The application reads environment variables; it does not automatically load `.env` files. Safiron additionally requires `OPENAI_API_KEY`, even when the Main or Guard model uses DeepSeek.

Provider presets in the CLI are `openai`, `deepseek`, and `deepseek-flash`. Use `-m` to override the Main model and `-gm` to override the Guard model with a full provider-qualified model ID. Use `-gp` to select a separate Guard provider.

## Quick start

Run a household task:

```powershell
python agent.py --embodied --benign-env --require-planning -p deepseek-flash "Put the book on the table."
```

Check the active skill registry and run with Guard:

```powershell
python GuardAgent/agent.py --list-stages
python agent.py --embodied --benign-env --require-planning --guard -p deepseek-flash "Put the book on the table."
```

The checked-in active configuration contains **AgentSpec** (`tool_selection`), **ParseData** (`tool_observation`), and **recover**. Safiron is available in the skill library for the `planning` stage. Enabling `--guard` checks only registered stages. See [Safety skills](docs/safety-skills.md) for activation, alternatives, and Safiron's independent detector.

## Evaluation

The `household-v2-45` benign dataset contains 45 standalone tasks across nine household families. Each task starts from a reset scene and is scored against explicit final-state criteria.

```powershell
python agent.py --benign --start 1 --num 45 -p deepseek-flash --save-results
python agent.py --benign --start 1 --num 45 -p deepseek-flash --guard --require-planning --save-results
python scripts/summarize_result.py result/<your_run>.json
```

Task definitions and offline reference actions are described in the [dataset catalog](data/benign/README.md). References and scoring goals are not passed to the Main Agent. The repository also includes stage-oriented attack evaluation datasets and runners; see [Evaluation and results](docs/evaluation.md) for CLI options, artifacts, and interpretation.

## Repository layout

| Path | Purpose |
| --- | --- |
| `agent.py` | Main Agent CLI and batch runner |
| `GuardAgent/` | Guard CLI, runtime, worker, and stage registry |
| `GuardAgent/skills/` | Active safety skills |
| `GuardAgent/skill_library/` | Available skill implementations and rule resources |
| `blackboard.py`, `message_provenance.py`, `stage_capture.py` | Stage snapshots, user-input binding, and middleware |
| `guard_*.py` | Guard communication, payloads, prefilters, and recovery |
| `embodied_env/` | Simulator, tools, task definitions, and scoring |
| `data/` | Evaluation prompts and memory datasets |
| `scripts/` | Dataset export, result summaries, and diagnostic runners |
| `tests/` | Offline regression and integration tests |
| `docs/` | Configuration, runtime, evaluation, and contributor documentation |
| `result/` | Historical tracked results and local generated runs |
| `deepagent/` | Compatibility copies of agent helpers; root modules serve the main CLI |

## Development

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
python -m pip check
```

Tests use scripted models and mocked detector responses, with simulator and transport integration checks. They do not establish live model safety or task-completion rates. See [Contributing](CONTRIBUTING.md) for validation and repository conventions.

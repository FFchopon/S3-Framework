# Contributing

Use Python 3.12, install `requirements-dev.txt`, and run commands from the repository root. Root-level helper modules are used by `agent.py`; if changing helpers that also exist in `deepagent/`, check both copies for compatibility.

## Validation

```powershell
python -m pytest tests -q
python -m pip check
python agent.py --help
python GuardAgent/agent.py --list-stages
git diff --check
```

Tests must run without live model requests. Safiron tests construct an isolated skill registry instead of requiring a particular local active configuration. A real model run is a separate experiment and must preserve its configuration and artifacts.

When editing benign definitions, run `python scripts/regenerate_benign_tasks.py` and include both exported JSON files. Keep task IDs, goals, reference trajectories, and dataset version consistent. Do not expose reference actions or scoring goals in agent-facing prompts.

When adding a safety skill, declare `name`, `description`, and `stage` in `SKILL.md`. Keep one active skill per stage. Add module metadata only for interpreter-backed skills. Update the skill catalog and test its real invocation path and error behavior.

## Preparing a GitHub submission

Include source, active skills and library resources, datasets, tests, dependency manifests, and documentation. Keep virtual environments, caches, credentials, generated run logs, and local skill backups outside the submission. `.gitignore` excludes new local artifacts; files already tracked by Git remain tracked.

Review `git status --short` and `git diff --stat` before staging. Historical tracked result files are retained in this checkout; select a separate, documented artifact bundle if publishing new experiment results. Do not treat historical audits as validation of the current runtime. The [implementation audit](docs/development/implementation-audit-2026-09-29.md) records a dated maintenance check.

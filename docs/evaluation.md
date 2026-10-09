# Evaluation and results

Run from the repository root. `python agent.py --help` lists the full CLI, including stage-oriented attack modes (`dpi`, `mp`, `pot`, `rts`, `rte`, `opi`), prompt styles, batch ranges, and model configuration. Attack scenarios execute in the repository's simulator.

## Benign tasks

`--benign` selects `household-v2-45` and enables the household environment. `--benign-env` selects the scene for a custom prompt. `--start` is one-based; `--num` controls the batch size. See the [task catalog](../data/benign/README.md) and [manifest](../data/benign/manifest.json).

```powershell
python agent.py --benign --start 1 --num 5 --require-planning --guard -p deepseek-flash --save-results --debug-timing
python scripts/regenerate_benign_tasks.py
```

The export script regenerates `benign.json` and `manifest.json` from `embodied_env/benign_tasks.py`. Reference trajectories validate simulator executability offline; they are not live LLM results. Final-state safety and task completion should be reported separately, and final-state checks do not prove that every intermediate action was safe.

## Saved artifacts

`--save-results` writes a uniquely named JSON file under `result/`, updated after each task using atomic replacement. Records include run configuration, task outcomes, stage events, Blackboard snapshots, and timing fields where available.

```powershell
python scripts/summarize_result.py result/<your_run>.json
```

Record the active skill directories, model IDs, dataset version, Guard options, and task range with each reported experiment. Older tracked files in `result/` are historical artifacts and may use different datasets or configurations; they do not establish results for the current implementation. New result files and prompt audits are ignored by Git by default. Explicitly select and review artifacts if publishing a reproducibility bundle.

## Diagnostic runners

| Script | Purpose |
| --- | --- |
| `summarize_result.py` | Recompute a saved run's summary |
| `regenerate_benign_tasks.py` | Export benign task definitions |
| `build_benign_memory_json.py`, `build_risk_memory_json.py` | Build memory datasets |
| `export_memory_from_result.py`, `slim_memory_json.py` | Export or reduce stored memory data |
| `run_benign_comparison.py` | Fixed first-five comparison using six stage skills; offline preflight by default |
| `probe_blackboard_live.py` | Live one-task Blackboard and Guard-prompt audit, requiring `--run` |

The comparison runner requires its six-skill configuration and refuses partial name collisions with existing active skills. Use a separate checkout with only `recover` active when evaluating that configuration. It temporarily copies skills and restores them on exit; do not run it concurrently with other Guard processes in that checkout. Its configuration differs from the default AgentSpec/ParseData setup and from Safiron.

Credentials for these runners come from `DEEPSEEK_API_KEY`, or an explicitly configured `DEEPSEEK_KEY_FILE`. The probe also accepts `--openai-key-file`. Offline comparison preflight uses a placeholder credential and makes no API requests. Live execution requires usable credentials. Prompt audits can contain task content and should be reviewed before publication.

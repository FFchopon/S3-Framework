# Result artifacts

This directory contains historical tracked experiment outputs and locally generated runs. New outputs are ignored by Git by default; historical tracked files remain in version control.

Use `python scripts/summarize_result.py result/<run>.json` to inspect a saved run. Interpret each file against its recorded model, dataset version, skill configuration, task range, and Guard options. Historical runs may not match the current code or `household-v2-45` corpus.

For current result fields and interpretation, see [Evaluation and results](../docs/evaluation.md). Review prompt traces and logs before explicitly adding a reproducibility bundle.

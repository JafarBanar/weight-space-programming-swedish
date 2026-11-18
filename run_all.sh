#!/usr/bin/env bash
# Reproduce everything. Cached stages in results/ are skipped; delete results/ to recompute from scratch.
set -euo pipefail
PY=${PY:-.venv/bin/python}
$PY src/fetch_data.py                    # UD text for perplexity (downloaded once)
$PY src/run_experiment.py                # weight reading, dev tuning, held-out test (main result)
$PY src/run_notices.py                   # post-hoc: factual accuracy on notice questions
$PY src/run_posthoc.py                   # post-hoc: delete each of the two neurons separately
$PY src/run_lora.py                      # conventional fine-tuning baseline
$PY src/robustness_weight_reading.py     # does the weight reading depend on the word list?
$PY src/analyze.py                       # tables, figures, examples

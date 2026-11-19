#!/usr/bin/env bash
# Round 2: confirmation on fresh prompts, survey across models, larger steering search.
set -uo pipefail
PY=${PY:-.venv/bin/python}
mkdir -p logs results/r2
$PY src/download_models.py                                   # extra open models (pinned revisions)
for m in qwen2.5-1.5b qwen2.5-0.5b eurollm-1.7b qwen3-1.7b; do
  $PY -u src/run_round2.py $m 2>&1 | grep --line-buffered -v -E "Warning|Loading weights|deprecated|Redirects" > logs/r2_$m.log
done
$PY -u src/run_round2.py qwen2.5-1.5b steering 2>&1 | grep --line-buffered -v -E "Warning|Loading weights|deprecated|Redirects" > logs/r2_steering.log
echo R2_DONE > logs/r2.done

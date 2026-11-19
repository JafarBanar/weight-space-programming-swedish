#!/usr/bin/env bash
# Round 3: find a fix that works. Usage: ./run_round3.sh [dev|test|all] [model]
# "dev" tunes every method on train/dev questions and writes frozen.json; "test" runs the frozen methods once.
set -uo pipefail
PY=${PY:-.venv/bin/python}
STAGE=${1:-all}; M=${2:-qwen2.5-1.5b}
mkdir -p logs results/r3
run() {  # run <log name> <args...>
  local log=$1; shift
  $PY -u src/run_round3.py "$M" "$@" 2>&1 | grep --line-buffered -v -E "Warning|Loading weights|deprecated|Redirects" > "logs/r3_${M}_${log}.log"
}
if [ "$STAGE" = dev ] || [ "$STAGE" = all ]; then
  run prep prep
  if [ "$M" = qwen2.5-1.5b ]; then                 # primary model: tune on dev
    run p3 p3
    run e1 e1
    run train_F2 train F2
    run train_F3a train F3 2e-4
    run train_F3b train F3 1e-3
    run loradev loradev
  else                                             # replication model: same choices as the primary, no retuning
    run e1 e1
    run train_F2 train F2
    run train_F3 train F3
  fi
  run freeze freeze
fi
if [ "$STAGE" = test ] || [ "$STAGE" = all ]; then
  run test test
fi
echo "R3_${STAGE}_DONE" > "logs/r3_${M}_${STAGE}.done"

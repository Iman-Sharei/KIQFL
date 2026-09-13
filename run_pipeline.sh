#!/usr/bin/env bash
# Full multi-suite KIQFL experiment pipeline (~18–25h on a gaming GPU).
# Usage: bash run_pipeline.sh
set -euo pipefail
cd "$(dirname "$0")"

if [[ -x .venv/bin/python ]]; then
  PY=.venv/bin/python
else
  PY=python3
fi

mkdir -p result
if [[ -f result/experiments_log.csv ]]; then
  stamp=$(date +%Y%m%d_%H%M%S)
  mv result/experiments_log.csv "result/experiments_log_pre_pipeline_${stamp}.csv"
fi

echo "=== KIQFL pipeline ==="
$PY -m core pipeline "$@"
echo "DONE. See result/results_summary.txt and result/"

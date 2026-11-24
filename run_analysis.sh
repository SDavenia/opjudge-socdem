#!/usr/bin/env bash
# Runs every analysis of one or more datasets into analysis/<dataset>/<setup>/ (joint = hb + vp pooled).
#   ./run_analysis.sh                # hb vp joint
#   ./run_analysis.sh vp joint
# Single-side setups first: analyze_both.py reads their signed gaps for its replication table.
set -euo pipefail
cd "$(dirname "$0")"
VARIABLE=${VARIABLE:-gender}
[ $# -eq 0 ] && set -- hb vp joint

scores() {  # scores <dataset> <setup>
  if [ "$1" = joint ]; then echo "scores/hb_$2 scores/vp_$2"; else echo "scores/$1_$2"; fi
}

for d in "$@"; do
  echo "=== $d ==="
  for s in reference_$VARIABLE candidate_$VARIABLE; do
    for r in "" --relative; do
      python analyze.py $(scores "$d" "$s") --out_dir "analysis/$d/$s" $r
    done
  done
  for r in "" --relative; do
    python analyze_both.py $(scores "$d" both_$VARIABLE) --out_dir "analysis/$d/both_$VARIABLE" \
      --single_side_dir "analysis/$d" $r
  done
  mkdir -p "analysis/$d/candidate_$VARIABLE/lmm"
  Rscript lmm/candidate.R "$(scores "$d" candidate_$VARIABLE | tr ' ' ',')" "analysis/$d/candidate_$VARIABLE/lmm" \
    2>&1 | tee "analysis/$d/candidate_$VARIABLE/lmm/log.txt"
done

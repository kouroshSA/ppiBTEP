#!/usr/bin/env bash
# Composite / ensemble figures for ppiBTEP across the V3 replicates (default 1 2),
# per condition. Staged as LES_V3-1..N; figures self-report n.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-/home/ksa/anaconda3/envs/gpt/bin/python}"
export MPLBACKEND=Agg
STAGEROOT="$(mktemp -d)"
LR="$ROOT/LES_Results_V3"
OUTROOT="$LR/ensemble"
MODELS=("$@"); [ ${#MODELS[@]} -eq 0 ] && MODELS=(1 2)
N=${#MODELS[@]}

comp () {  # $1=cond
  local cond="$1"; local stage; stage="$(mktemp -d -p "$STAGEROOT")"
  local i=1
  for k in "${MODELS[@]}"; do ln -s "$LR/V3-$k/LES_$cond" "$stage/LES_V3-$i"; i=$((i+1)); done
  local out="$OUTROOT/$cond"; mkdir -p "$out"
  "$PY" "$ROOT/make_composite_les.py" --parent "$stage" --run_glob 'LES_V3-*' \
      --out "$out" --pos_col Probability_Friends
  "$PY" "$ROOT/composite_roc_btep.py" --parent "$stage" --models "$N" \
      --model-name ppiBTEP --out "$out/ROC"
}

echo "Composite across ${N} models: V3-${MODELS[*]}"
for cond in regular no_homodimers ps1_random ps2_random ps1-ps2_random; do
  echo "===== $cond ====="; comp "$cond"
done
rm -rf "$STAGEROOT"
echo "ALL_BTEP_COMPOSITES_DONE -> $OUTROOT"

#!/usr/bin/env bash
# eval_per_epoch.sh — per-epoch PRS/RRS evaluation for ppiBTEP V3 training runs.
#
# Instead of scoring every checkpoint at the end of a run, this scores each epoch
# checkpoint ONCE as soon as it appears (across the six reference-set conditions),
# then assembles that condition's summary_table.csv / trajectories / ROC / violins
# with LES-wrapper --skip_inference (no re-inference). Per-epoch results are thus
# available right after each epoch, and the score-once design means no checkpoint
# is inferred twice. Output layout matches run_les_all_conditions.sh, so
# run_composites_all_conditions.sh consumes it unchanged.
#
# Usage:
#   ./eval_per_epoch.sh [k ...]            # watch replicates (default: 1 2)
#   ONCE=1 ./eval_per_epoch.sh 1 2 3       # single pass over existing checkpoints, then exit
#
# Config via environment (all optional):
#   PY        python interpreter          (default: esm env)
#   REF       reference-set root          (default: <repo>/V3_PRS-RRS)
#   RESULTS   dir holding results_V3-k/   (default: <repo>)
#   OUT       output root                 (default: <repo>/LES_Results_V3)
#   MODEL_CONFIG, NUM_LAYERS, MAX_LENGTH  (defaults: esm1b, 6, 1024)
#   INTERVAL  poll seconds                (default: 30)
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-/home/ksa/anaconda3/envs/esm/bin/python}"
REF="${REF:-$ROOT/V3_PRS-RRS}"
RESULTS="${RESULTS:-$ROOT}"
OUT="${OUT:-$ROOT/LES_Results_V3}"
MODEL_CONFIG="${MODEL_CONFIG:-facebook/esm1b_t33_650M_UR50S}"
NUM_LAYERS="${NUM_LAYERS:-6}"; MAX_LENGTH="${MAX_LENGTH:-1024}"; INTERVAL="${INTERVAL:-30}"
INFER="$ROOT/inference_ppiBTPE_2GPU.py"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" MPLBACKEND=Agg PYTHONUNBUFFERED=1

REG="$REF/PRS-RRS"; NH="$REF/PRS-RRS_no_homodimers"; HO="$REF/PRS-RRS_homodimers_only"; RC="$REF/random_controls"
CONDS=(regular no_homodimers homodimers_only ps1_random ps2_random ps1-ps2_random)
MODELS=("$@"); [ ${#MODELS[@]} -eq 0 ] && MODELS=(1 2)

prs_for(){ case "$1" in regular) echo "$REG/PRS-V3-$2.csv";; no_homodimers) echo "$NH/PRS-V3-$2.csv";;
  homodimers_only) echo "$HO/PRS-V3-$2.csv";;
  ps1_random) echo "$RC/PRS-V3-${2}_ps1_random.csv";; ps2_random) echo "$RC/PRS-V3-${2}_ps2_random.csv";;
  ps1-ps2_random) echo "$RC/PRS-V3-${2}_ps1-ps2_random.csv";; esac; }
rrs_for(){ case "$1" in regular) echo "$REG/RRS-V3-$2.csv";; no_homodimers) echo "$NH/RRS-V3-$2.csv";;
  homodimers_only) echo "$HO/RRS-V3-$2.csv";;
  ps1_random) echo "$RC/RRS-V3-${2}_ps1_random.csv";; ps2_random) echo "$RC/RRS-V3-${2}_ps2_random.csv";;
  ps1-ps2_random) echo "$RC/RRS-V3-${2}_ps1-ps2_random.csv";; esac; }

stable(){ local a b; a=$(stat -c%s "$1" 2>/dev/null||echo 0); sleep 3; b=$(stat -c%s "$1" 2>/dev/null||echo 0); [ "$a" = "$b" ] && [ "$a" != 0 ]; }

score_ckpt(){ # k epoch_str ckpt_path
  local k="$1" es="$2" cp="$3" cond P R out pc rc
  for cond in "${CONDS[@]}"; do
    P="$(prs_for "$cond" "$k")"; R="$(rrs_for "$cond" "$k")"
    [ -f "$P" ] && [ -f "$R" ] || continue
    out="$OUT/V3-$k/LES_$cond/epoch_$es"; mkdir -p "$out"
    pc="$out/PRS_epoch${es}_probabilities.csv"; rc="$out/RRS_epoch${es}_probabilities.csv"
    [ -s "$pc" ] || "$PY" "$INFER" --model_path "$cp" --num_layers "$NUM_LAYERS" --max_length "$MAX_LENGTH" \
        --model_config "$MODEL_CONFIG" --input_file "$P" --output_file "$pc" --device cuda >/dev/null 2>&1
    [ -s "$rc" ] || "$PY" "$INFER" --model_path "$cp" --num_layers "$NUM_LAYERS" --max_length "$MAX_LENGTH" \
        --model_config "$MODEL_CONFIG" --input_file "$R" --output_file "$rc" --device cuda >/dev/null 2>&1
  done
}

assemble(){ # k  — build per-condition summaries/trajectories from existing CSVs (no GPU)
  local k="$1" incf="" cond P R out
  [ -f "$RESULTS/results_V3-$k/model/ppiBTPE_final.pth" ] && incf="--include_final"
  for cond in "${CONDS[@]}"; do
    out="$OUT/V3-$k/LES_$cond"; [ -d "$out" ] || continue
    P="$(prs_for "$cond" "$k")"; R="$(rrs_for "$cond" "$k")"
    "$PY" "$ROOT/LES-wrapper.py" --checkpoint_dir "$RESULTS/results_V3-$k/model" \
        --num_layers "$NUM_LAYERS" --max_length "$MAX_LENGTH" $incf --skip_inference \
        --model_config "$MODEL_CONFIG" --prs_file "$P" --rrs_file "$R" --output_dir "$out" >/dev/null 2>&1 \
      || echo "  [warn] assemble V3-$k $cond"
  done
}

model_done(){ # k -> true if final.pth exists and its last condition is scored
  [ -f "$RESULTS/results_V3-$k/model/ppiBTPE_final.pth" ] || return 1
  [ -s "$OUT/V3-$k/LES_${CONDS[-1]}/epoch_final/PRS_epochfinal_probabilities.csv" ]
}

echo "[$(date '+%F %T')] eval_per_epoch start  models=${MODELS[*]}  ref=$REF"
while true; do
  progressed=0; alldone=1
  for k in "${MODELS[@]}"; do
    MODEL="$RESULTS/results_V3-$k/model"; [ -d "$MODEL" ] || { alldone=0; continue; }
    newk=0
    for cp in "$MODEL"/ppiBTPE_epoch_*.pth "$MODEL"/ppiBTPE_final.pth; do
      [ -f "$cp" ] || continue
      es="$(basename "$cp" | sed -E 's/ppiBTPE_epoch_([0-9]+)\.pth/\1/; s/ppiBTPE_final\.pth/final/')"
      # marker = last condition's PRS csv (scored last) => epoch fully done
      [ -s "$OUT/V3-$k/LES_${CONDS[-1]}/epoch_$es/PRS_epoch${es}_probabilities.csv" ] && continue
      stable "$cp" || continue
      score_ckpt "$k" "$es" "$cp"; newk=1; progressed=1
      echo "[$(date '+%F %T')] scored V3-$k epoch $es"
    done
    [ "$newk" = 1 ] && { assemble "$k"; echo "[$(date '+%F %T')] assembled V3-$k"; }
    model_done "$k" || alldone=0
  done
  [ "${ONCE:-0}" = 1 ] && { echo "[$(date '+%F %T')] ONCE pass complete"; break; }
  [ "$alldone" = 1 ] && { echo "[$(date '+%F %T')] all requested models complete — exit"; break; }
  sleep "$INTERVAL"
done

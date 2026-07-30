#!/usr/bin/env bash
# LES for ppiBTEP V3-1 and V3-2 across five reference-set conditions, reusing the
# ppiDCE reference / control sets (identical 2-column format). ppiBTEP V3 config:
# num_layers 6, max_length 1024, esm1b; all 10 epoch checkpoints + final. The three
# random-control conditions auto-skip AUC/F1 (no true positives). Resumable.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-/home/ksa/anaconda3/envs/esm/bin/python}"
export HF_HUB_OFFLINE=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

REFROOT=/home/ksa/Models/ppiDCE
REG="$REFROOT/MED4_PRS-RRS"
NH="$REFROOT/MED4_PRS-RRS_no_homodimers"
RC="$REFROOT/PRS-RRS_random_controls"
OUTROOT="$ROOT/LES_Results_V3"
MODELS=("$@"); [ ${#MODELS[@]} -eq 0 ] && MODELS=(1 2)

les () {  # $1=label $2=prs $3=rrs $4=ckptdir $5=outdir
  [ -f "$5/summary_table.csv" ] && { echo "  $1 done -- skip"; return; }
  for f in "$2" "$3"; do [ -f "$f" ] || { echo "  MISSING $f -- skip $1"; return; }; done
  echo "######## V3-$k  $1  $(date '+%F %T') ########"
  "$PY" "$ROOT/LES-wrapper.py" \
      --checkpoint_dir "$4" --num_layers 6 --max_length 1024 --include_final \
      --model_config facebook/esm1b_t33_650M_UR50S \
      --prs_file "$2" --rrs_file "$3" --output_dir "$5"
  echo "=== V3-$k $1 summary ==="; cat "$5/summary_table.csv"
}

for k in "${MODELS[@]}"; do
  CK="$ROOT/results_V3-${k}/model"
  [ -f "$CK/ppiBTPE_final.pth" ] || { echo "MISSING checkpoints for V3-$k -- skip"; continue; }
  O="$OUTROOT/V3-$k"
  les regular        "$REG/PRS-V3-$k.csv"                "$REG/RRS-V3-$k.csv"                "$CK" "$O/LES_regular"
  les no_homodimers  "$NH/PRS-V3-$k.csv"                 "$NH/RRS-V3-$k.csv"                 "$CK" "$O/LES_no_homodimers"
  les ps1_random     "$RC/PRS-V3-${k}_ps1_random.csv"     "$RC/RRS-V3-${k}_ps1_random.csv"     "$CK" "$O/LES_ps1_random"
  les ps2_random     "$RC/PRS-V3-${k}_ps2_random.csv"     "$RC/RRS-V3-${k}_ps2_random.csv"     "$CK" "$O/LES_ps2_random"
  les ps1-ps2_random "$RC/PRS-V3-${k}_ps1-ps2_random.csv" "$RC/RRS-V3-${k}_ps1-ps2_random.csv" "$CK" "$O/LES_ps1-ps2_random"
done
echo "ALL_BTEP_LES_DONE  $(date '+%F %T')"

#!/usr/bin/env bash
# Stage 4-C (third pass, closure only): retrain r70_30 so its event log is complete.
# Why: the 18:24 run produced a valid 150-step adapter, but the 18:49 relaunch overwrote
# out/train_r70_30.jsonl and then died at step 12 - so the training-side numbers for that
# config were lost twice (stdout buffer + overwritten events). The old adapter is preserved
# under models/_backup_r70_30_1824 and the partial log under out/train_r70_30_partial_1856.jsonl.
set -uo pipefail
cd "$(dirname "$0")"
PY=.venv/Scripts/python.exe
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
mkdir -p out logs

USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
if [ "${USED:-0}" -gt 4000 ]; then
  echo "!!! refusing to start: GPU already using ${USED} MiB" | tee -a logs/stage4.log
  exit 9
fi
echo "=== stage4c run start $(date '+%F %T') gpu_used=${USED}MiB ===" | tee -a logs/stage4.log

echo "--- train r70_30 start $(date '+%F %T')" | tee -a logs/stage4.log
$PY -Xutf8 -u scripts/04_train_lora.py --config r70_30 --steps 150 --accum 8 --max_len 512 \
    > logs/train_r70_30.log 2>&1
rc=$?
echo "--- train r70_30 rc=$rc done $(date '+%F %T')" | tee -a logs/stage4.log
[ $rc -ne 0 ] && tail -20 logs/train_r70_30.log | tee -a logs/stage4.log

if [ -f models/lora_r70_30/adapter_model.safetensors ]; then
  echo "--- eval lora_r70_30 start $(date '+%F %T')" | tee -a logs/stage4.log
  $PY -Xutf8 -u scripts/05_eval.py --adapter models/lora_r70_30 > logs/eval_lora_r70_30.log 2>&1
  echo "--- eval lora_r70_30 rc=$? done $(date '+%F %T')" | tee -a logs/stage4.log
else
  echo "--- eval skipped: adapter missing after rc=$rc" | tee -a logs/stage4.log
fi

echo "=== stage4c run end $(date '+%F %T') ===" | tee -a logs/stage4.log

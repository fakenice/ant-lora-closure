#!/usr/bin/env bash
# Stage 4-C (second pass): train ONLY the two ratio configs that the previous runs lost
# (r30_70 aborted by the slow-step guard, r0_100 aborted too). r70_30 (18:24) and r100_0 (18:49)
# already hold valid 150-step adapters and must not be touched.
# Then Stage 4-D: evaluate base + the four adapters on the 100-item deterministic MCQ set.
#
# Two hardening measures vs. the last attempt:
#   1. one loop only - abort immediately if the GPU already looks busy (a training process
#      from a previous, stopped background task would still hold >4 GiB);
#   2. python -u so step progress reaches the log live instead of dying inside a stdout buffer.
set -uo pipefail
cd "$(dirname "$0")"
PY=.venv/Scripts/python.exe
STEPS=${STEPS:-150}
ACCUM=${ACCUM:-8}
MAXLEN=${MAXLEN:-512}
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
mkdir -p out logs

USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
if [ "${USED:-0}" -gt 4000 ]; then
  echo "!!! refusing to start: GPU already using ${USED} MiB - another training process is alive." \
      | tee -a logs/stage4.log
  exit 9
fi

echo "=== stage4b run start $(date '+%F %T') steps=$STEPS accum=$ACCUM max_len=$MAXLEN gpu_used=${USED}MiB ===" \
  | tee -a logs/stage4.log

for cfg in r30_70 r0_100; do
  echo "--- train $cfg start $(date '+%F %T')" | tee -a logs/stage4.log
  $PY -Xutf8 -u scripts/04_train_lora.py --config "$cfg" --steps "$STEPS" --accum "$ACCUM" \
      --max_len "$MAXLEN" > "logs/train_${cfg}.log" 2>&1
  rc=$?
  echo "--- train $cfg rc=$rc done $(date '+%F %T')" | tee -a logs/stage4.log
  if [ $rc -ne 0 ]; then
    echo "!!! train $cfg FAILED; tail of log:" | tee -a logs/stage4.log
    tail -20 "logs/train_${cfg}.log" | tee -a logs/stage4.log
  fi
done

for tag in base models/lora_r100_0 models/lora_r70_30 models/lora_r30_70 models/lora_r0_100; do
  name=$(basename "$tag")
  if [ "$tag" != "base" ] && [ ! -f "$tag/adapter_model.safetensors" ]; then
    echo "--- skip eval $name (adapter missing)" | tee -a logs/stage4.log
    continue
  fi
  echo "--- eval $name start $(date '+%F %T')" | tee -a logs/stage4.log
  $PY -Xutf8 -u scripts/05_eval.py --adapter "$tag" > "logs/eval_${name}.log" 2>&1
  echo "--- eval $name rc=$? done $(date '+%F %T')" | tee -a logs/stage4.log
done

echo "=== stage4b run end $(date '+%F %T') ===" | tee -a logs/stage4.log

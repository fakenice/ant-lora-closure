#!/usr/bin/env bash
# Stage 4-C: train the four data-ratio configs serially (one GPU, 8 cores - no overlap),
# then Stage 4-D: evaluate base + the four adapters on the 100-item deterministic MCQ set.
# Everything is append-logged under out/ so a failure is diagnosable after the fact.
set -uo pipefail
cd "$(dirname "$0")"
PY=.venv/Scripts/python.exe
STEPS=${STEPS:-150}
ACCUM=${ACCUM:-8}
MAXLEN=${MAXLEN:-512}
# The 2080 Ti is shared with the desktop session; allocator fragmentation plus WDDM
# spill into shared system memory is the failure mode observed on the first attempt.
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
mkdir -p out logs
echo "=== stage4 run start $(date '+%F %T') steps=$STEPS accum=$ACCUM max_len=$MAXLEN ===" | tee -a logs/stage4.log

for cfg in r100_0 r70_30 r30_70 r0_100; do
  echo "--- train $cfg start $(date '+%F %T')" | tee -a logs/stage4.log
  $PY -Xutf8 scripts/04_train_lora.py --config "$cfg" --steps "$STEPS" --accum "$ACCUM" \
      --max_len "$MAXLEN" > "logs/train_${cfg}.log" 2>&1
  rc=$?
  echo "--- train $cfg rc=$rc done $(date '+%F %T')" | tee -a logs/stage4.log
  if [ $rc -ne 0 ]; then
    echo "!!! train $cfg FAILED; tail of log:" | tee -a logs/stage4.log
    tail -25 "logs/train_${cfg}.log" | tee -a logs/stage4.log
  fi
done

for tag in base models/lora_r100_0 models/lora_r70_30 models/lora_r30_70 models/lora_r0_100; do
  name=$(basename "$tag")
  if [ "$tag" != "base" ] && [ ! -f "$tag/adapter_model.safetensors" ]; then
    echo "--- skip eval $name (adapter missing)" | tee -a logs/stage4.log
    continue
  fi
  echo "--- eval $name start $(date '+%F %T')" | tee -a logs/stage4.log
  $PY -Xutf8 scripts/05_eval.py --adapter "$tag" > "logs/eval_${name}.log" 2>&1
  echo "--- eval $name rc=$? done $(date '+%F %T')" | tee -a logs/stage4.log
done

echo "=== stage4 run end $(date '+%F %T') ===" | tee -a logs/stage4.log

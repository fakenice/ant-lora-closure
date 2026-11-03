#!/usr/bin/env bash
# 4 ratio configs × 3 training seeds on one GPU, then evaluate base + every adapter
# on the leak-filtered 500-item set. Serial on purpose: the 11 GB card cannot host
# two training loops (see README §6.9).
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
STEPS=${STEPS:-150}
ACCUM=${ACCUM:-8}
MAXLEN=${MAXLEN:-512}
SEEDS=${SEEDS:-"20261014 20261015 20261016"}
CONFIGS="r100_0 r70_30 r30_70 r0_100"
EVAL_FILE=data/clean/mcq_eval_500.jsonl
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
mkdir -p out logs out/eval

echo "=== multiseed start $(date '+%F %T') seeds='$SEEDS' steps=$STEPS ===" | tee -a logs/multiseed_500.log

# Refuse to start while a local LLM is holding VRAM.
if ollama ps 2>/dev/null | grep -q "qwen3"; then
  echo "!!! Ollama still holds a model; run: ollama stop qwen3:14b" | tee -a logs/multiseed_500.log
  exit 9
fi

check_vram() {
  $PY - <<'EOF'
import sys, torch
free, total = torch.cuda.mem_get_info()
free_mib = free / 2**20
print(f"free VRAM {free_mib:.0f} MiB / {total/2**20:.0f} MiB")
sys.exit(0 if free_mib > 8000 else 9)
EOF
}

for seed in $SEEDS; do
  for cfg in $CONFIGS; do
    tag="${cfg}_s${seed}"
    if [ -f "models/lora_${tag}/adapter_model.safetensors" ]; then
      echo "--- skip train $tag (adapter exists)" | tee -a logs/multiseed_500.log
      continue
    fi
    if ! check_vram >> logs/multiseed_500.log 2>&1; then
      echo "!!! low VRAM before $tag; stopping" | tee -a logs/multiseed_500.log
      exit 9
    fi
    echo "--- train $tag start $(date '+%F %T')" | tee -a logs/multiseed_500.log
    $PY -Xutf8 scripts/04_train_lora.py --config "$cfg" --seed "$seed" --tag "$tag" \
        --steps "$STEPS" --accum "$ACCUM" --max_len "$MAXLEN" \
        > "logs/train_${tag}.log" 2>&1
    rc=$?
    echo "--- train $tag rc=$rc done $(date '+%F %T')" | tee -a logs/multiseed_500.log
    if [ $rc -ne 0 ]; then
      echo "!!! train $tag failed; tail:" | tee -a logs/multiseed_500.log
      tail -20 "logs/train_${tag}.log" | tee -a logs/multiseed_500.log
    fi
  done
done

eval_one() {
  local adapter="$1" tag="$2"
  if [ -f "out/eval/eval_${tag}.json" ]; then
    echo "--- skip eval $tag (exists)" | tee -a logs/multiseed_500.log
    return 0
  fi
  if [ "$adapter" != "base" ] && [ ! -f "${adapter}/adapter_model.safetensors" ]; then
    echo "--- skip eval $tag (adapter missing)" | tee -a logs/multiseed_500.log
    return 0
  fi
  echo "--- eval $tag start $(date '+%F %T')" | tee -a logs/multiseed_500.log
  $PY -Xutf8 scripts/05_eval.py --adapter "$adapter" --eval-file "$EVAL_FILE" --tag "$tag" \
      > "logs/eval_${tag}.log" 2>&1
  echo "--- eval $tag rc=$? done $(date '+%F %T')" | tee -a logs/multiseed_500.log
}

eval_one base base_500
for seed in $SEEDS; do
  for cfg in $CONFIGS; do
    tag="${cfg}_s${seed}"
    eval_one "models/lora_${tag}" "${tag}_500"
  done
done

$PY -Xutf8 scripts/07_multiseed_summary.py 2>&1 | tee -a logs/multiseed_500.log
echo "=== multiseed end $(date '+%F %T') ===" | tee -a logs/multiseed_500.log

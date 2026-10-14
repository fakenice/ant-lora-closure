#!/bin/sh
set -eux
cd "$(dirname "$0")"
uv venv --python 3.11 .venv
uv pip install -p .venv/Scripts/python.exe "torch==2.6.0" --index-url https://download.pytorch.org/whl/cu124
uv pip install -p .venv/Scripts/python.exe transformers peft datasets accelerate datasketch tqdm
.venv/Scripts/python.exe -c "import torch,transformers,peft,datasets;print('torch',torch.__version__,'cuda',torch.cuda.is_available());print('transformers',transformers.__version__,'peft',peft.__version__,'datasets',datasets.__version__)"

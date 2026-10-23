#!/bin/bash
# serve_vllm.sh — serve the frozen backbone through an OpenAI-compatible endpoint.
# Usage: bash serve_vllm.sh            (defaults: Qwen/Qwen3-8B on port 8000, 8192-token window)
#        MODEL=Qwen/Qwen3-8B PORT=8000 MAX_MODEL_LEN=8192 bash serve_vllm.sh
set -euo pipefail
MODEL="${MODEL:-Qwen/Qwen3-8B}"
PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
python -c "import vllm" 2>/dev/null || { echo "vllm is not installed: pip install vllm" >&2; exit 2; }
exec vllm serve "$MODEL" \
    --host 127.0.0.1 --port "$PORT" \
    --tensor-parallel-size "$TP_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --enable-prefix-caching

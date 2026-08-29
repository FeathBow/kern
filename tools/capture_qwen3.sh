#!/usr/bin/env bash
# Mine Qwen3-4B kernels out of vLLM: one prompt, one output token, with the
# CUPTI injection lib (vendored from pegainfer PR #982) recording every
# module's cubin and every launch's full call ABI into dumped-kernels/.
#
# Run on a tray with a free GPU:
#   ssh pod4-gb300-3-tray05-f3 'CUDA_VISIBLE_DEVICES=0 /mnt/shared/home/susun/kern/tools/capture_qwen3.sh'
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model="${MODEL:-/mnt/shared/weights/Qwen3-4B}"
out="${KERNEL_CAPTURE_DIR:-$repo/dumped-kernels}"

[ -f "$repo/tools/kernel-capture/libkernelcapture.so" ] || "$repo/tools/kernel-capture/build.sh"

mkdir -p "$out"
# FlashInfer JIT-builds its attention kernel with ninja + nvcc; both must be
# on PATH since we invoke the venv python directly rather than activating it.
export PATH="$repo/.venv/bin:${CUDA_HOME:-/usr/local/cuda}/bin:$PATH"
export CUDA_INJECTION64_PATH="$repo/tools/kernel-capture/libkernelcapture.so"
export KERNEL_CAPTURE_DIR="$out"
export HF_HUB_OFFLINE=1
export VLLM_NO_USAGE_STATS=1
# Run EngineCore in-process: the injection lib then sees the one pid that
# actually launches kernels, and its stderr is not swallowed by a subproc.
export VLLM_ENABLE_V1_MULTIPROCESSING=0

# enforce_eager: every launch is a real cuLaunchKernel with live staged
# params, and prefill/decode both surface without CUDA-graph replay opacity.
"$repo/.venv/bin/python" - "$model" <<'EOF'
import sys
from vllm import LLM, SamplingParams

llm = LLM(model=sys.argv[1], enforce_eager=True, gpu_memory_utilization=0.45,
          max_model_len=4096)
out = llm.generate(["hi"], SamplingParams(max_tokens=1))
print("output token:", repr(out[0].outputs[0].text))
EOF

echo "--- capture summary ---"
for d in "$out"/pid*/; do
  echo "$d: $(ls "$d" | grep -c cubin) cubins, $(wc -l < "$d/launches.jsonl") launches"
done

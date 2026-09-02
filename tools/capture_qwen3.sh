#!/usr/bin/env bash
# Mine Qwen3-4B kernels out of vLLM: one prompt, one output token, with the
# CUPTI injection lib (vendored from pegainfer PR #982) recording every
# module's cubin and every launch's full call ABI into dumped-kernels/.
#
# Run on a tray with a free GPU:
#   ssh <tray> 'CUDA_VISIBLE_DEVICES=0 $HOME/kern/tools/capture_qwen3.sh'
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
#
# Several prompts of increasing length, sent one at a time (bs=1): each is
# one prefill pass at a different token count, giving the grid-expression
# fitter samples of the same dispatch under different `tokens` values —
# `ceil_div(tokens, c)` vs `tokens` cannot be told apart from a single tiny
# prompt. Diverse prose only: repetitive filler makes the first sampled
# token EOS and yields an empty-output false positive.
"$repo/.venv/bin/python" - "$model" <<'EOF'
import sys
from vllm import LLM, SamplingParams
from vllm.config import AttentionConfig

# TRITON_ATTN：唯一 flat-ABI 的 attention backend（FA4/trtllm-gen 都是
# packed struct / TMA descriptor，不可 rebind，见 README）。0.28 起只能
# 用 AttentionConfig 选，VLLM_ATTENTION_BACKEND 环境变量已删除。
llm = LLM(model=sys.argv[1], enforce_eager=True, gpu_memory_utilization=0.45,
          max_model_len=4096,
          attention_config=AttentionConfig(backend="TRITON_ATTN"))

prompts = [
    "hi",
    "The harbor master kept a ledger of every ship that wintered in the bay, "
    "noting cargo, crew, and the state of each hull in a cramped, looping hand.",
    "When the observatory finally reopened after the renovation, the docents "
    "discovered that the old refractor had been quietly recollimated by a "
    "retired machinist who lived nearby. He left no note, only a small brass "
    "shim on the pier and a chalked arrow pointing at Polaris. The director "
    "considered filing a complaint, then looked through the eyepiece at Saturn "
    "and decided some trespasses are better rewarded than reported.",
    "The floodplain census took three summers to complete. In the first "
    "summer the crews mapped oxbow lakes and counted heron rookeries from "
    "canoes, losing two clipboards and one outboard motor to the river. In "
    "the second they walked transects through willow thickets, recording "
    "beaver sign, sediment depth, and the stranded hulks of fence posts from "
    "farms abandoned in the forties. The third summer was all reconciliation: "
    "duplicate plots resolved, disputed species calls sent to the herbarium, "
    "and the great argument over whether the channel had truly migrated or "
    "merely braided settled by an afternoon with the 1962 aerial photographs. "
    "The final report ran to four hundred pages, and the appendix everyone "
    "actually read — the one with the flood marks painted on grain elevators "
    "— was written in a single evening by the youngest technician on the crew.",
]
for p in prompts:
    out = llm.generate([p], SamplingParams(max_tokens=4))
    o = out[0]
    print(f"prompt_tokens={len(o.prompt_token_ids)} output={o.outputs[0].text!r}")
EOF

echo "--- capture summary ---"
for d in "$out"/pid*/; do
  echo "$d: $(ls "$d" | grep -c cubin) cubins, $(wc -l < "$d/launches.jsonl") launches"
done

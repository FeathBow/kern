#!/usr/bin/env bash
# Mine the DSpark speculative-decoding path: Qwen3-4B target + dspark 5-layer
# draft (deepseek-ai/dspark_qwen3_4b_block7), TRITON_ATTN, enforce_eager,
# greedy. Captures the draft backbone (non-causal unified instance), the
# fused context-KV precompute, and the target verify passes.
#
#   ssh <tray> 'CUDA_VISIBLE_DEVICES=0 /mnt/shared/home/susun/kern/tools/capture_qwen3_spec.sh'
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model="${MODEL:-/mnt/shared/weights/Qwen3-4B}"
draft="${DRAFT:-$repo/weights/dspark_qwen3_4b_block7}"
out="${KERNEL_CAPTURE_DIR:-$repo/dumped-kernels}"

[ -f "$repo/tools/kernel-capture/libkernelcapture.so" ] || "$repo/tools/kernel-capture/build.sh"

mkdir -p "$out"
export PATH="$repo/.venv/bin:${CUDA_HOME:-/usr/local/cuda}/bin:$PATH"
export CUDA_INJECTION64_PATH="$repo/tools/kernel-capture/libkernelcapture.so"
export KERNEL_CAPTURE_DIR="$out"
export HF_HUB_OFFLINE=1
export VLLM_NO_USAGE_STATS=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0

"$repo/.venv/bin/python" - "$model" "$draft" <<'EOF'
import sys
from vllm import LLM, SamplingParams
from vllm.config import AttentionConfig

llm = LLM(model=sys.argv[1], enforce_eager=True, gpu_memory_utilization=0.45,
          max_model_len=4096,
          attention_config=AttentionConfig(backend="TRITON_ATTN"),
          speculative_config={
              "model": sys.argv[2],
              "method": "dspark",
              "num_speculative_tokens": 7,
              "enable_adaptive_verification": False,
          })

prompts = [
    "The capital of France is",
    "The harbor master kept a ledger of every ship that wintered in the bay, "
    "noting cargo, crew, and the state of each hull in a cramped, looping hand.",
    "When the observatory finally reopened after the renovation, the docents "
    "discovered that the old refractor had been quietly recollimated by a "
    "retired machinist who lived nearby. He left no note, only a small brass "
    "shim on the pier and a chalked arrow pointing at Polaris. The director "
    "considered filing a complaint, then looked through the eyepiece at Saturn "
    "and decided some trespasses are better rewarded than reported.",
]
for p in prompts:
    out = llm.generate([p], SamplingParams(max_tokens=32, temperature=0.0))
    o = out[0]
    print(f"prompt_tokens={len(o.prompt_token_ids)} output={o.outputs[0].text!r}")

m = llm.llm_engine.get_metrics() if hasattr(llm.llm_engine, "get_metrics") else None
if m:
    for metric in m:
        if "spec" in metric.name or "accept" in metric.name:
            print(metric.name, getattr(metric, "value", getattr(metric, "sum", None)))
EOF

echo "--- capture summary ---"
for d in "$out"/pid*/; do
  echo "$d: $(ls "$d" | grep -c cubin) cubins, $(wc -l < "$d/launches.jsonl") launches"
done

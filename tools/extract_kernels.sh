#!/usr/bin/env bash
# 从 capture dump 的 module cubin 里抽出 manifest 用到的核，连同自编的
# embedding 核放进 kernels/。同名 Triton 核的多个 constexpr 实例会分散在
# 多个 module 里，全部抽出——runtime 加载时按 cuFuncGetParamInfo 的参数
# 布局与 manifest 声明比对来消歧（phase-2 校验兼做实例选择）。
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dump="${1:-$repo/dumped-kernels/pid3977275}"
out="$repo/kernels"
mkdir -p "$out"
rm -f "$out"/module_*.cubin

# manifest 里的全部 cubin 符号（extern: 前缀是 runtime 特判，跳过）
mapfile -t syms < <(python3 - "$repo/examples/qwen3-4b-decode.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
for k in m["kernels"].values():
    s = k["symbol"]
    if not s.startswith("extern:") and not s.startswith("kern_"):
        print(s)
PY
)

cuobjdump="${CUDA_HOME:-/usr/local/cuda}/bin/cuobjdump"
n=0
for mod in "$dump"/module_*.cubin; do
  names="$("$cuobjdump" -symbols "$mod" 2>/dev/null | awk '{print $NF}')" || continue
  for s in "${syms[@]}"; do
    if grep -qxF "$s" <<<"$names"; then
      cp "$mod" "$out/$(basename "$mod")"
      n=$((n+1))
      break
    fi
  done
done

for src in "$repo"/tools/kernels-src/*.cu; do
  nvcc -cubin -arch=sm_103a -o "$out/$(basename "$src" .cu).cubin" "$src"
done
echo "extracted $n modules + embedding.cubin -> $out"
ls -la "$out" | tail -n +2

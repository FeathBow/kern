#!/usr/bin/env bash
# 从 capture dump 的 module cubin 里抽出 manifest 用到的核，连同自编核
# （tools/kernels-src/*.cu）放进 kernel 目录。
#
#   tools/extract_kernels.sh <manifest.json> <dump_dir>[:<dump_dir>...] [out_dir=kernels]
#
# 两种取法：manifest 里钉定了 `cubin`（+sha256）的 step 按 sha 在所有 dump
# 目录（以及仓库 kernels/ 里借用的核，如 DSpark 的 unified_noncausal）中查
# 找，以 manifest 里的文件名落地（同名 Triton 核的多个 constexpr 实例 ABI
# 相同，只能靠钉定；跨 dump 时生成器按 sha 前缀命名，避免 module_N 撞名）；
# 没钉定的 symbol 把含它的 module 全部抽出，runtime 加载时按
# cuFuncGetParamInfo 的参数布局与 manifest 声明比对来消歧。不同模型的
# dump 别混进同一个 out_dir——同名同 ABI 的实例（如 reshape_and_cache 的
# block_size 16 / 784 版本）会串。
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
manifest="${1:?manifest.json}"
IFS=: read -r -a dumps <<<"${2:?dump dir(s)}"
out="${3:-$repo/kernels}"
mkdir -p "$out"
rm -f "$out"/module_*.cubin

# (pinned cubin, sha) pairs and unpinned symbols from the manifest
# (extern: is a runtime built-in, kern_ is a handwritten kernel)
mapfile -t pinned < <(python3 - "$manifest" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
seen = set()
for k in m["kernels"].values():
    for st in k["impl"]["steps"]:
        s = st["symbol"]
        if s.startswith("extern:") or s.startswith("kern_"):
            continue
        key = (st.get("cubin") or "-", st.get("sha256") or "-", s)
        if key not in seen:
            seen.add(key)
            print(*key)
PY
)

# content index over every candidate cubin (dumps + the repo's borrowed kernels)
declare -A by_sha
while read -r sha path; do
  by_sha[$sha]="${by_sha[$sha]:-$path}"
done < <(for dmp in "${dumps[@]}"; do sha256sum "$dmp"/*.cubin; done; sha256sum "$repo"/kernels/*.cubin 2>/dev/null)

cuobjdump="${CUDA_HOME:-/usr/local/cuda}/bin/cuobjdump"
n=0
declare -A copied
for entry in "${pinned[@]}"; do
  read -r cubin sha sym <<<"$entry"
  if [ "$cubin" != "-" ]; then
    [ -n "${copied[$cubin]:-}" ] && continue
    src=""
    if [ "$sha" != "-" ]; then
      src="${by_sha[$sha]:-}"
      [ -n "$src" ] || { echo "pinned $cubin ($sym) sha $sha not found in ${dumps[*]} / kernels" >&2; exit 1; }
    else
      for dmp in "${dumps[@]}"; do [ -f "$dmp/$cubin" ] && { src="$dmp/$cubin"; break; }; done
      [ -n "$src" ] || { echo "pinned $cubin ($sym) missing in ${dumps[*]}" >&2; exit 1; }
    fi
    [ "$(realpath "$src")" = "$(realpath -m "$out/$cubin")" ] || cp "$src" "$out/$cubin"
    copied[$cubin]=1; n=$((n+1))
  else
    for dmp in "${dumps[@]}"; do
      for mod in "$dmp"/module_*.cubin; do
        base="$(basename "$mod")"
        [ -n "${copied[$base]:-}" ] && continue
        names="$("$cuobjdump" -symbols "$mod" 2>/dev/null | awk '{print $NF}')" || continue
        if grep -qxF "$sym" <<<"$names"; then
          cp "$mod" "$out/$base"; copied[$base]=1; n=$((n+1))
        fi
      done
    done
  fi
done

h=0
for src in "$repo"/tools/kernels-src/*.cu; do
  nvcc -cubin -arch=sm_103a -o "$out/$(basename "$src" .cu).cubin" "$src"
  h=$((h+1))
done
echo "extracted $n modules + $h handwritten cubins -> $out"
ls "$out" | tr '\n' ' '; echo

#!/usr/bin/env bash
# nvcc the Kimi-K3 decode kernel families into cubins.
#
#   tools/build_k3_kernels.sh [out_dir=target/cubins]
#     KERN_SM=sm_103a  KERN_REBUILD=1
#     TILELANG_SRC=<tilelang/src>  CUTLASS_INCLUDE=<cutlass/include>   (the kernel-lab
#     container has both under /usr/local/lib/python3.12/dist-packages/tilelang)
#
# Sources are pegainfer's AOT TileLang K3 kernels, vendored under
# tools/k3-tilelang (pegainfer-k3/kernels/generate.py output — see the README
# there). Each family compiles to one cubin holding every batch bucket; the
# two state kernels additionally get line-indexed wrappers
# (tools/k3_line_shim.py) so a sequence's state can live in a kern
# `bytes_per_seq` line. Flags are pegainfer's own (build.rs): -O3, C++20, no
# fast-math. The generators pin the cubins by sha256.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out="${1:-$repo/target/cubins}"
arch="${KERN_SM:-sm_103a}"
tl="$repo/tools/k3-tilelang"
tmpl="${TILELANG_SRC:-/usr/local/lib/python3.12/dist-packages/tilelang/src}"
cutlass="${CUTLASS_INCLUDE:-/usr/local/lib/python3.12/dist-packages/tilelang/3rdparty/cutlass/include}"
[ -d "$tmpl/tl_templates" ] || { echo "TILELANG_SRC=$tmpl has no tl_templates (run inside kernel-lab or point TILELANG_SRC at tilelang/src)" >&2; exit 1; }
mkdir -p "$out"
flags=(-cubin "-arch=$arch" -O3 --std=c++20 -w -Xcudafe --diag_suppress=177 "-I$tmpl" "-I$cutlass")
n=0
build() {  # build <src.cu> <dst.cubin>
  if [ -z "${KERN_REBUILD:-}" ] && [ "$2" -nt "$1" ]; then return; fi
  nvcc "${flags[@]}" -o "$2" "$1"
  n=$((n+1))
}
for f in rms_norm_rbs land land_rms_norm_rbs add2 mul_sigmoid situ router_topk attnres_scores attnres_mix; do
  build "$tl/k3_${f}_batched.cu" "$out/k3_tl_${f}.cubin"
done
for f in kda_core conv_silu; do
  shim="$out/k3_tl_${f}_line.cu"
  if [ -n "${KERN_REBUILD:-}" ] || [ ! "$shim" -nt "$tl/k3_${f}_batched.cu" ] || [ ! "$shim" -nt "$repo/tools/k3_line_shim.py" ]; then
    python3 "$repo/tools/k3_line_shim.py" "$tl/k3_${f}_batched.cu" "$shim"
  fi
  build "$shim" "$out/k3_tl_${f}_line.cubin"
done
echo "built $n K3 TileLang cubins ($arch) -> $out" >&2

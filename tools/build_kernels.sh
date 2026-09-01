#!/usr/bin/env bash
# nvcc every handwritten kernel (tools/kernels-src/*.cu) into one directory.
#
#   tools/build_kernels.sh [out_dir=target/cubins]      KERN_SM=sm_103a  KERN_REBUILD=1
#
# The generators pin these builds by sha256 (tools/handwritten.py), and
# extract_kernels.sh copies them into the kernel dir by that sha. nvcc output
# is deterministic for a given compiler — a different nvcc is a different
# cubin, and the manifest pins whichever build was present when it was
# generated (kern-test will tell you whether the two agree numerically).
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out="${1:-$repo/target/cubins}"
arch="${KERN_SM:-sm_103a}"
mkdir -p "$out"
n=0
for src in "$repo"/tools/kernels-src/*.cu; do
  dst="$out/$(basename "$src" .cu).cubin"
  if [ -z "${KERN_REBUILD:-}" ] && [ "$dst" -nt "$src" ]; then continue; fi
  nvcc -cubin -arch="$arch" -o "$dst" "$src"
  n=$((n+1))
done
echo "built $n handwritten cubins ($arch) -> $out" >&2

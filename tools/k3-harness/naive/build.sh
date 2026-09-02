#!/usr/bin/env bash
# Build every naive kernel to naive/<name>.cubin.
#   ./build.sh            # all
#   ./build.sh kda_core   # one
# `_pre.txt` is the shared preamble the sources were generated from; it is not
# a source file (each .cu already contains it inline and is self-contained).
set -euo pipefail
cd "$(dirname "$0")"
NVCC=${NVCC:-/usr/local/cuda-13.1/bin/nvcc}
ARCH=${ARCH:-sm_103a}
pick=("$@")
for f in k3_*.cu; do
  name=${f#k3_}; name=${name%.cu}
  if [ ${#pick[@]} -gt 0 ]; then
    hit=0; for p in "${pick[@]}"; do [ "$p" = "$name" ] && hit=1; done
    [ $hit -eq 1 ] || continue
  fi
  echo "  nvcc -cubin $f -> $name.cubin"
  "$NVCC" -cubin -arch=$ARCH -O3 -o "$name.cubin" "$f"
done
echo "naive cubins built."

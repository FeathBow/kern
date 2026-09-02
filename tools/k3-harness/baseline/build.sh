#!/usr/bin/env bash
# Build the retired pegainfer K5 kernel (the timing baseline K5 must beat 3x).
set -euo pipefail
cd "$(dirname "$0")"
NVCC=${NVCC:-/usr/local/cuda-13.1/bin/nvcc}
ARCH=${ARCH:-sm_103a}
"$NVCC" -cubin -arch=$ARCH -O3 -o mla_paged_attn_old.cubin k3_mla_paged_attn_old.cu
echo "baseline/mla_paged_attn_old.cubin built."

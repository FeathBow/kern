// Kimi-K3 paged latent append: one step's post-norm kv latent (512) and rope
// half (64) become the 576-wide bf16 row of each token's slot in this MLA
// layer's slice of its page.
//
//   kern_k3_kv_append(kv_norm, rope, slot_mapping, slab, layer_off, page_stride, tokens)
//     grid tokens, block 288 (one bf16 pair per thread)
//
// Slab layout (elements): page p at p * page_stride, layer slice at
// layer_off, token t of the page at t * 576. A token slot s (the runtime's
// `slot_mapping`, 64 slots per page) lives at page s / 64, row s % 64 —
// the same addressing the attention kernel's page walk uses through the
// block table.
//
//   nvcc -cubin -arch=sm_103a -o kernels/k3_kv_append.cubin tools/kernels-src/k3_kv_append.cu
#include <cuda_bf16.h>

extern "C" __global__ void kern_k3_kv_append(
    const __nv_bfloat162* __restrict__ kv_norm,   // [tokens, 512] as pairs
    const __nv_bfloat162* __restrict__ rope,      // [tokens, 64] as pairs
    const long long* __restrict__ slot_mapping,   // [tokens]
    __nv_bfloat162* __restrict__ slab,            // pool base, as pairs
    long long layer_off,                          // elements
    long long page_stride,                        // elements
    int tokens) {
  const int t = blockIdx.x;
  if (t >= tokens) return;
  const long long slot = slot_mapping[t];
  const long long row = (slot / 64) * page_stride + layer_off + (slot % 64) * 576;
  __nv_bfloat162* dst = slab + row / 2;
  const int j = threadIdx.x;  // 0..287 pairs
  if (j < 256) {
    dst[j] = kv_norm[(long long)t * 256 + j];
  } else if (j < 288) {
    dst[j] = rope[(long long)t * 32 + (j - 256)];
  }
}

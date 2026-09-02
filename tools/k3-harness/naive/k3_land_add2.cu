#include <cuda_bf16.h>

typedef __nv_bfloat16 bf16;
#define K3_H        7168
#define K3_HEADS      96
#define K3_INNER   12288
#define K3_NB_MAX      8
#define K3_KDA_FUSED 49152
#define K3_MLA_FUSED 14400
#define K3_Q_LORA   1536
#define K3_KV_LORA   512
#define K3_ROPE       64
#define K3_ROW       576
#define K3_Q_B     18432
#define K3_WSM       256
#define K3_EXPERTS   224
#define K3_TOPK       16
#define K3_PAGE       64
#define K3_EPS     1e-5f
#define K3_LB      -5.0f
#define K3_REC_BYTES 6291456LL
#define K3_WIN_BYTES   73728LL

__device__ __forceinline__ float b2f(bf16 x) { return __bfloat162float(x); }
__device__ __forceinline__ bf16 f2b(float x) { return __float2bfloat16_rn(x); }
__device__ __forceinline__ float sigf(float x) { return 1.0f / (1.0f + expf(-x)); }

// [K1c] naive reference transcription — one block per row.
//   grid (B,1,1)  block (1024,1,1)  dynamic smem 0
// hidden = bf16( prefix2 + bf16(p1) + (two ? bf16(p2) : 0) )
// `two == 0` is the dense layer: p2 is a valid pointer but is not read.

extern "C" __global__ void kern_k3_land_add2(
    const float* __restrict__ p1, const float* __restrict__ p2,
    const bf16* __restrict__ prefix2, bf16* __restrict__ hidden, int two,
    int B) {
  const int b = blockIdx.x;
  if (b >= B) return;
  for (int i = threadIdx.x; i < K3_H; i += blockDim.x) {
    const size_t k = (size_t)b * K3_H + i;
    float a = b2f(prefix2[k]) + b2f(f2b(p1[k]));
    if (two) a += b2f(f2b(p2[k]));
    hidden[k] = f2b(a);
  }
}

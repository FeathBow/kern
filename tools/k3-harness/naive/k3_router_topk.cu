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

// [K6] naive reference transcription — one block per row.
//   grid (B,1,1)  block (256,1,1)  dynamic smem 0
// sig = sigma(S); biased = sig + bias; 16 sequential max scans, tie -> small e.

extern "C" __global__ void kern_k3_router_topk(
    const float* __restrict__ S, const float* __restrict__ bias,
    const bf16* __restrict__ rs, int* __restrict__ idx, float* __restrict__ wts,
    int B) {
  const int b = blockIdx.x;
  if (b >= B) return;
  __shared__ float sig[K3_EXPERTS], biased[K3_EXPERTS];
  __shared__ int pick[K3_TOPK];
  __shared__ float denom;
  for (int e = threadIdx.x; e < K3_EXPERTS; e += blockDim.x) {
    const float s = sigf(S[(size_t)b * K3_EXPERTS + e]);
    sig[e] = s;
    biased[e] = s + bias[e];
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float sum = 0.0f;
    for (int t = 0; t < K3_TOPK; ++t) {
      int best = 0;
      float bv = biased[0];
      for (int e = 1; e < K3_EXPERTS; ++e)
        if (biased[e] > bv) { bv = biased[e]; best = e; }   // strict: tie -> small e
      pick[t] = best;
      sum += sig[best];
      biased[best] = -1e30f;
    }
    denom = sum + 1e-20f;
  }
  __syncthreads();
  const float scale = b2f(rs[0]);
  for (int t = threadIdx.x; t < K3_TOPK; t += blockDim.x) {
    idx[(size_t)b * K3_TOPK + t] = pick[t];
    wts[(size_t)b * K3_TOPK + t] = sig[pick[t]] / denom * scale;
  }
}

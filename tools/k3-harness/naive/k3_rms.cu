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

// [K6] naive generic rms — one block per row.
//   grid (B,1,1)  block (1024,1,1)  dynamic smem 0

__device__ __forceinline__ float blk_sum(float v, float* red) {
  __syncthreads();
  for (int off = 16; off; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
  const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
  if (l == 0) red[w] = v;
  __syncthreads();
  if (threadIdx.x == 0) {
    float s = 0.0f;
    for (int i = 0; i < (int)(blockDim.x >> 5); ++i) s += red[i];
    red[31] = s;
  }
  __syncthreads();
  return red[31];
}

extern "C" __global__ void kern_k3_rms(const bf16* __restrict__ x,
                                       const bf16* __restrict__ gamma,
                                       bf16* __restrict__ o, int h, int B) {
  const int b = blockIdx.x;
  if (b >= B) return;
  __shared__ float red[32];
  float ss = 0.0f;
  for (int i = threadIdx.x; i < h; i += blockDim.x) {
    const float v = b2f(x[(size_t)b * h + i]);
    ss += v * v;
  }
  ss = blk_sum(ss, red);
  const float r = rsqrtf(ss / (float)h + K3_EPS);
  for (int i = threadIdx.x; i < h; i += blockDim.x) {
    const bf16 y = f2b(b2f(x[(size_t)b * h + i]) * r);
    o[(size_t)b * h + i] = __hmul(y, gamma[i]);
  }
}

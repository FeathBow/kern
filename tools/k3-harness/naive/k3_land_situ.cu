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

// [K7] naive land_situ — one thread per output element.
//   grid (B, ceil(n/1024), 1)  block (1024,1,1)  dynamic smem 0
// situ(g,u) = 4*tanh(g/4)*sigma(g) * 25*tanh(u/25); gate in the first n
// columns, up in the next n.

extern "C" __global__ void kern_k3_land_situ(const float* __restrict__ p,
                                             bf16* __restrict__ act, int n,
                                             int B) {
  const int b = blockIdx.x;
  const int i = blockIdx.y * blockDim.x + threadIdx.x;
  if (b >= B || i >= n) return;
  const float g = b2f(f2b(p[(size_t)b * 2 * n + i]));
  const float u = b2f(f2b(p[(size_t)b * 2 * n + n + i]));
  act[(size_t)b * n + i] = f2b(4.0f * tanhf(g * 0.25f) * sigf(g) * 25.0f *
                               tanhf(u * 0.04f));
}

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

// [K4] naive reference transcription — one block per row.
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

extern "C" __global__ void kern_k3_mla_prep(
    const float* __restrict__ partial, const bf16* __restrict__ gamma_q_a,
    const bf16* __restrict__ gamma_kv_a, const long long* __restrict__ slot_mapping,
    bf16* __restrict__ slab, long long layer_off, long long page_stride,
    bf16* __restrict__ q_norm, bf16* __restrict__ mla_gate, int B) {
  const int b = blockIdx.x;
  if (b >= B) return;
  const int tid = threadIdx.x, nt = blockDim.x;
  __shared__ float red[32];
  const float* P = partial + (size_t)b * K3_MLA_FUSED;

  float ss = 0.0f;
  for (int i = tid; i < K3_Q_LORA; i += nt) { float v = b2f(f2b(P[i])); ss += v * v; }
  ss = blk_sum(ss, red);
  float r = rsqrtf(ss / (float)K3_Q_LORA + K3_EPS);
  for (int i = tid; i < K3_Q_LORA; i += nt) {
    bf16 y = f2b(b2f(f2b(P[i])) * r);
    q_norm[(size_t)b * K3_Q_LORA + i] = __hmul(y, gamma_q_a[i]);
  }

  ss = 0.0f;
  for (int i = tid; i < K3_KV_LORA; i += nt) {
    float v = b2f(f2b(P[K3_Q_LORA + i]));
    ss += v * v;
  }
  ss = blk_sum(ss, red);
  r = rsqrtf(ss / (float)K3_KV_LORA + K3_EPS);

  const long long slot = slot_mapping[b];
  bf16* row = slab + (slot / K3_PAGE) * page_stride + layer_off +
              (slot % K3_PAGE) * (long long)K3_ROW;
  for (int i = tid; i < K3_KV_LORA; i += nt) {
    bf16 y = f2b(b2f(f2b(P[K3_Q_LORA + i])) * r);
    row[i] = __hmul(y, gamma_kv_a[i]);
  }
  for (int i = tid; i < K3_ROPE; i += nt)
    row[K3_KV_LORA + i] = f2b(P[K3_Q_LORA + K3_KV_LORA + i]);
  for (int i = tid; i < K3_INNER; i += nt)
    mla_gate[(size_t)b * K3_INNER + i] = f2b(P[K3_Q_LORA + K3_ROW + i]);
}

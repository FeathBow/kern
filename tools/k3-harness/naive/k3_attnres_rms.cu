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

// [K1a] naive reference transcription — one block per row, 1024 threads.
//   grid (B,1,1)  block (1024,1,1)  dynamic smem 0 (all shared is static)
// mixed = attnres(blocks, prefix, nb); if (snapshot) blocks[b,nb] = prefix;
// normed = rms(mixed, gamma)

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

extern "C" __global__ void kern_k3_attnres_rms(
    const bf16* __restrict__ prefix, bf16* __restrict__ blocks,
    const float* __restrict__ sw, const bf16* __restrict__ gamma,
    bf16* __restrict__ normed, int nb, int snapshot, int B) {
  const int b = blockIdx.x;
  if (b >= B) return;
  const int tid = threadIdx.x, nt = blockDim.x;
  __shared__ float red[32];
  __shared__ float pr[K3_NB_MAX + 1];
  __shared__ bf16 mixed[K3_H];

  const bf16* pref = prefix + (size_t)b * K3_H;
  const bf16* blk = blocks + (size_t)b * K3_NB_MAX * K3_H;
  const int nc = nb + 1;

  if (nb > 0) {
    for (int c = 0; c < nc; ++c) {
      const bf16* cand = (c < nb) ? (blk + (size_t)c * K3_H) : pref;
      float ss = 0.0f;
      for (int i = tid; i < K3_H; i += nt) { float v = b2f(cand[i]); ss += v * v; }
      ss = blk_sum(ss, red);
      const float r = rsqrtf(ss / (float)K3_H + K3_EPS);
      float dot = 0.0f;
      for (int i = tid; i < K3_H; i += nt) dot += b2f(cand[i]) * r * sw[i];
      dot = blk_sum(dot, red);
      if (tid == 0) pr[c] = dot;
      __syncthreads();
    }
    if (tid == 0) {
      float m = pr[0];
      for (int c = 1; c < nc; ++c) m = fmaxf(m, pr[c]);
      float l = 0.0f;
      for (int c = 0; c < nc; ++c) { pr[c] = expf(pr[c] - m); l += pr[c]; }
      for (int c = 0; c < nc; ++c) pr[c] /= l;
    }
    __syncthreads();
  }

  for (int i = tid; i < K3_H; i += nt) {
    if (nb == 0) {
      mixed[i] = pref[i];
    } else {
      float acc = 0.0f;
      for (int c = 0; c < nc; ++c) {
        const bf16* cand = (c < nb) ? (blk + (size_t)c * K3_H) : pref;
        acc += pr[c] * b2f(cand[i]);
      }
      mixed[i] = f2b(acc);
    }
  }
  __syncthreads();

  float ss = 0.0f;
  for (int i = tid; i < K3_H; i += nt) { float v = b2f(mixed[i]); ss += v * v; }
  ss = blk_sum(ss, red);
  const float r = rsqrtf(ss / (float)K3_H + K3_EPS);
  for (int i = tid; i < K3_H; i += nt) {
    bf16 y = f2b(b2f(mixed[i]) * r);
    normed[(size_t)b * K3_H + i] = __hmul(y, gamma[i]);
  }
  if (snapshot) {
    __syncthreads();
    bf16* dst = blocks + ((size_t)b * K3_NB_MAX + nb) * K3_H;
    for (int i = tid; i < K3_H; i += nt) dst[i] = pref[i];
  }
}

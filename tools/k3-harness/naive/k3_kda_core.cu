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

// [K3] naive reference transcription — one block per (row, head), thread = dv.
//   grid (B, HEADS, 1)  block (128,1,1)  dynamic smem 0

__device__ __forceinline__ float blk_sum128(float v, float* red) {
  __syncthreads();
  for (int off = 16; off; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
  const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
  if (l == 0) red[w] = v;
  __syncthreads();
  if (threadIdx.x == 0) {
    float s = 0.0f;
    for (int i = 0; i < (int)(blockDim.x >> 5); ++i) s += red[i];
    red[7] = s;
  }
  __syncthreads();
  return red[7];
}

extern "C" __global__ void kern_k3_kda_core(
    const bf16* __restrict__ conv_q, const bf16* __restrict__ conv_k,
    const bf16* __restrict__ conv_v, const float* __restrict__ wsm_partial,
    const float* __restrict__ gate_partial, const bf16* __restrict__ w_f_b,
    const float* __restrict__ dt_bias, const float* __restrict__ a_log,
    const float* __restrict__ gamma_o, void* __restrict__ kda_base,
    const int* __restrict__ line_index, long long line_bytes,
    bf16* __restrict__ out, int B) {
  const int b = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
  if (b >= B) return;
  __shared__ float red[8];
  __shared__ float qs[128], kn[128], dec[128];
  __shared__ bf16 flow[128], attn[128];

  const size_t hoff = (size_t)b * K3_INNER + (size_t)h * 128;
  const bf16 q = conv_q[hoff + d], k = conv_k[hoff + d], v = conv_v[hoff + d];

  float qtot = blk_sum128(b2f(__hmul(q, q)), red);
  float ktot = blk_sum128(b2f(__hmul(k, k)), red);
  const bf16 qr = f2b(rsqrtf(b2f(f2b(qtot)) + 1e-6f));
  const bf16 kr = f2b(rsqrtf(b2f(f2b(ktot)) + 1e-6f));
  qs[d] = b2f(__hmul(q, qr)) * 0.08838834764831845f;  // 128^-0.5
  kn[d] = b2f(__hmul(k, kr));

  const float beta = sigf(b2f(f2b(wsm_partial[(size_t)b * K3_WSM + h])));
  flow[d] = f2b(wsm_partial[(size_t)b * K3_WSM + 96 + d]);
  __syncthreads();

  float ga = 0.0f;
  for (int j = 0; j < 128; ++j)
    ga += b2f(flow[j]) * b2f(w_f_b[((size_t)h * 128 + d) * 128 + j]);
  const float raw = b2f(f2b(ga)) + dt_bias[(size_t)h * 128 + d];
  dec[d] = expf(K3_LB * sigf(expf(a_log[h]) * raw));
  __syncthreads();

  float* S = (float*)((char*)kda_base + (long long)line_index[b] * line_bytes) +
             ((size_t)h * 128 + d) * 128;
  float m = 0.0f;
  for (int j = 0; j < 128; ++j) m += S[j] * dec[j] * kn[j];
  const float dlt = (b2f(v) - m) * beta;
  float a = 0.0f;
  for (int j = 0; j < 128; ++j) {
    const float sp = S[j] * dec[j] + dlt * kn[j];
    S[j] = sp;
    a += sp * qs[j];
  }
  attn[d] = f2b(a);
  __syncthreads();

  const float av = b2f(attn[d]);
  const float ss = blk_sum128(av * av, red);
  const float r = rsqrtf(ss / 128.0f + K3_EPS);
  const bf16 o = f2b(av * r * gamma_o[d]);
  const float g = b2f(f2b(gate_partial[(size_t)b * K3_KDA_FUSED + 3 * K3_INNER +
                                       (size_t)h * 128 + d]));
  out[hoff + d] = __hmul(o, f2b(sigf(g)));
}

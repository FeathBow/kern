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

// [K5] naive reference transcription — one block per (row, head).
//   grid (B, 96, 1)  block (128,1,1)  dynamic smem 0
// Three passes over the context (max, denominator, attend); a page's landed
// scores are recomputed each pass so no context-sized scratch is needed.
// No cleverness: this is the correctness/timing floor, not a target.

__device__ __forceinline__ float blk_max128(float v, float* red) {
  __syncthreads();
  for (int off = 16; off; off >>= 1)
    v = fmaxf(v, __shfl_down_sync(0xffffffff, v, off));
  const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
  if (l == 0) red[w] = v;
  __syncthreads();
  if (threadIdx.x == 0) {
    float s = red[0];
    for (int i = 1; i < (int)(blockDim.x >> 5); ++i) s = fmaxf(s, red[i]);
    red[7] = s;
  }
  __syncthreads();
  return red[7];
}
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

extern "C" __global__ void kern_k3_mla_paged_attn(
    const float* __restrict__ q_partial, const bf16* __restrict__ w_kv_b,
    const bf16* __restrict__ cache, const int* __restrict__ block_table,
    int max_pages, long long page_stride, const int* __restrict__ seq_lens,
    const bf16* __restrict__ scale, const bf16* __restrict__ mla_gate,
    bf16* __restrict__ gated, int B) {
  const int b = blockIdx.x, h = blockIdx.y;
  if (b >= B) return;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int nwarps = blockDim.x >> 5;
  __shared__ bf16 qabs[K3_ROW];
  __shared__ float sc[K3_PAGE];
  __shared__ float red[8];
  __shared__ bf16 lat[K3_KV_LORA];

  const float* qp = q_partial + (size_t)b * K3_Q_B + (size_t)h * 192;
  const bf16* w_uk = w_kv_b + (size_t)h * 256 * K3_KV_LORA;
  const bf16* w_uv = w_kv_b + ((size_t)h * 256 + 128) * K3_KV_LORA;

  for (int j = tid; j < K3_KV_LORA; j += blockDim.x) {
    float acc = 0.0f;
    for (int d = 0; d < 128; ++d)
      acc += b2f(f2b(qp[d])) * b2f(w_uk[(size_t)d * K3_KV_LORA + j]);
    qabs[j] = f2b(acc);
  }
  for (int j = tid; j < K3_ROPE; j += blockDim.x) qabs[K3_KV_LORA + j] = f2b(qp[128 + j]);
  __syncthreads();

  const int n = seq_lens[b];
  const int npg = (n + K3_PAGE - 1) / K3_PAGE;
  const bf16 sca = scale[0];

  // one warp per token; lanes stride the 576 dims; bf16 landing then bf16 scale
#define K3_PAGE_SCORES(cp, len)                                              \
  do {                                                                       \
    __syncthreads();                                                         \
    for (int t = warp; t < (len); t += nwarps) {                             \
      const bf16* row = (cp) + (size_t)t * K3_ROW;                           \
      float a = 0.0f;                                                        \
      for (int d = lane; d < K3_ROW; d += 32) a += b2f(qabs[d]) * b2f(row[d]);\
      for (int off = 16; off; off >>= 1) a += __shfl_down_sync(0xffffffff, a, off); \
      if (lane == 0) sc[t] = b2f(__hmul(f2b(a), sca));                        \
    }                                                                        \
    __syncthreads();                                                         \
  } while (0)

  float lm = -1e30f;
  for (int pg = 0; pg < npg; ++pg) {
    const bf16* cp = cache + (long long)block_table[(size_t)b * max_pages + pg] * page_stride;
    const int len = min(K3_PAGE, n - pg * K3_PAGE);
    K3_PAGE_SCORES(cp, len);
    for (int t = tid; t < len; t += blockDim.x) lm = fmaxf(lm, sc[t]);
  }
  const float m = blk_max128(lm, red);

  float ls = 0.0f;
  for (int pg = 0; pg < npg; ++pg) {
    const bf16* cp = cache + (long long)block_table[(size_t)b * max_pages + pg] * page_stride;
    const int len = min(K3_PAGE, n - pg * K3_PAGE);
    K3_PAGE_SCORES(cp, len);
    for (int t = tid; t < len; t += blockDim.x) ls += expf(sc[t] - m);
  }
  const float l = blk_sum128(ls, red);

  float oacc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int pg = 0; pg < npg; ++pg) {
    const bf16* cp = cache + (long long)block_table[(size_t)b * max_pages + pg] * page_stride;
    const int len = min(K3_PAGE, n - pg * K3_PAGE);
    K3_PAGE_SCORES(cp, len);
    for (int t = tid; t < len; t += blockDim.x) sc[t] = b2f(f2b(expf(sc[t] - m) / l));
    __syncthreads();
    for (int i = 0; i < 4; ++i) {
      const int j = i * 128 + tid;
      float a = oacc[i];
      for (int t = 0; t < len; ++t) a += sc[t] * b2f(cp[(size_t)t * K3_ROW + j]);
      oacc[i] = a;
    }
  }
#undef K3_PAGE_SCORES
  __syncthreads();
  for (int i = 0; i < 4; ++i) lat[i * 128 + tid] = f2b(oacc[i]);
  __syncthreads();

  const int dv = tid;
  float acc = 0.0f;
  for (int j = 0; j < K3_KV_LORA; ++j)
    acc += b2f(w_uv[(size_t)dv * K3_KV_LORA + j]) * b2f(lat[j]);
  const size_t k = (size_t)b * K3_INNER + (size_t)h * 128 + dv;
  gated[k] = __hmul(f2b(acc), f2b(sigf(b2f(mla_gate[k]))));
}

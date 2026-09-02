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

// [K2] naive reference transcription — one thread per (row, stream, column).
//   grid (B, 3, 24)  block (128,1,1)  dynamic smem 0   (4 columns per thread)
// The column loop is a grid-stride over INNER, so any (gridDim.z, blockDim.x)
// whose product divides INNER works — including the older (B,3,48)/256.
// y = sum_{t<3} f32(win[t][c])*cw[s][t][c] + f32(x)*cw[s][3][c];
// out = bf16(sb*sigma(sb)); the window shifts by one tap (tap 0 oldest).

extern "C" __global__ void kern_k3_conv_silu(
    const float* __restrict__ partial, const float* __restrict__ cw,
    void* __restrict__ kda_base, const int* __restrict__ line_index,
    long long line_bytes, bf16* __restrict__ conv_q, bf16* __restrict__ conv_k,
    bf16* __restrict__ conv_v, int B) {
  const int b = blockIdx.x;
  const int s = blockIdx.y;
  if (b >= B) return;

  bf16* win = (bf16*)((char*)kda_base + (long long)line_index[b] * line_bytes +
                      K3_REC_BYTES + (long long)s * K3_WIN_BYTES);
  bf16* out = (s == 0) ? conv_q : (s == 1) ? conv_k : conv_v;
  const int cstride = gridDim.z * blockDim.x;
  for (int c = blockIdx.z * blockDim.x + threadIdx.x; c < K3_INNER; c += cstride) {
    const bf16 x = f2b(partial[(size_t)b * K3_KDA_FUSED + (size_t)s * K3_INNER + c]);
    float y = 0.0f;
    for (int t = 0; t < 3; ++t)
      y += b2f(win[(size_t)t * K3_INNER + c]) * cw[((size_t)s * 4 + t) * K3_INNER + c];
    y += b2f(x) * cw[((size_t)s * 4 + 3) * K3_INNER + c];
    const bf16 sb = f2b(y);
    const float sv = b2f(sb);
    out[(size_t)b * K3_INNER + c] = f2b(sv * sigf(sv));

    win[(size_t)0 * K3_INNER + c] = win[(size_t)1 * K3_INNER + c];
    win[(size_t)1 * K3_INNER + c] = win[(size_t)2 * K3_INNER + c];
    win[(size_t)2 * K3_INNER + c] = x;
  }
}

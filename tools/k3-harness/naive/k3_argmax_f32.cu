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

// [K6] naive argmax over f32 logits, two stages, tie -> smallest index.
//   partial: grid (B, 64, 1)  block (1024,1,1)  dynamic smem 0
//   final:   grid (B, 1, 1)   block (64,1,1)    dynamic smem 0
// The partial split is a grid-stride over the row: part p owns
// i = p*blockDim.x + tid, striding gridDim.y*blockDim.x.

__device__ __forceinline__ void reduce_argmax(float* sv, int* si, int nt) {
  for (int s = nt >> 1; s > 0; s >>= 1) {
    if ((int)threadIdx.x < s) {
      const int j = threadIdx.x + s;
      if (sv[j] > sv[threadIdx.x] ||
          (sv[j] == sv[threadIdx.x] && si[j] < si[threadIdx.x])) {
        sv[threadIdx.x] = sv[j];
        si[threadIdx.x] = si[j];
      }
    }
    __syncthreads();
  }
}

extern "C" __global__ void kern_k3_argmax_f32_partial(
    const float* __restrict__ logits, float* __restrict__ pmax,
    int* __restrict__ pidx, int n) {
  __shared__ float sv[1024];
  __shared__ int si[1024];
  const int b = blockIdx.x, part = blockIdx.y, parts = gridDim.y;
  const int nt = blockDim.x, tid = threadIdx.x;
  const float* row = logits + (size_t)b * n;
  float bv = -3.402823466e+38f;
  int bi = 0x7fffffff;
  for (int i = part * nt + tid; i < n; i += parts * nt) {
    const float v = row[i];
    if (v > bv || (v == bv && i < bi)) { bv = v; bi = i; }
  }
  sv[tid] = bv;
  si[tid] = bi;
  __syncthreads();
  reduce_argmax(sv, si, nt);
  if (tid == 0) {
    pmax[(size_t)b * parts + part] = sv[0];
    pidx[(size_t)b * parts + part] = si[0];
  }
}

extern "C" __global__ void kern_k3_argmax_f32_final(
    const float* __restrict__ pmax, const int* __restrict__ pidx,
    long long* __restrict__ out, int parts) {
  __shared__ float sv[1024];
  __shared__ int si[1024];
  const int b = blockIdx.x, nt = blockDim.x, tid = threadIdx.x;
  float bv = -3.402823466e+38f;
  int bi = 0x7fffffff;
  for (int p = tid; p < parts; p += nt) {
    const float v = pmax[(size_t)b * parts + p];
    const int i = pidx[(size_t)b * parts + p];
    if (v > bv || (v == bv && i < bi)) { bv = v; bi = i; }
  }
  sv[tid] = bv;
  si[tid] = bi;
  __syncthreads();
  reduce_argmax(sv, si, nt);
  if (tid == 0) out[b] = (long long)si[0];
}

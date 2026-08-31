// DFlash2 candidate selector: rank-256 bilinear edge scores between the
// previous step's chosen candidate and this step's 16 candidates, then the
// greedy 7-step walk (vLLM qwen3_dflash2.CandidateSelector + the
// _selector_walk_kernel with temperature 0).
//
//   pred_r = predecessor_codebook[prev token]      (step 0: the anchor)
//   s[c]   = unary[l, c] + bf16( sum_r bf16(pred_r[r] * hidden_r[l, r]) * succ[l, c, r] )
//   idx    = argmax_c s[c]  (first max);  token[l] = cand[l, idx]
//
// Inputs are the gathered codebook rows: `succ` = successor rows of every
// candidate [steps, 16, rank]; `pred` = predecessor rows of every candidate
// [steps, 16, rank] (used as the predecessor of the *next* step);
// `pred_anchor` = predecessor row of the anchor [rank].  One block of
// `rank` threads: thread r owns rank lane r, the 16 dot products are block
// reductions.  Rounding follows vLLM's bf16 chain (elementwise product to
// bf16, matmul accumulated in f32 and stored bf16, added to the f32 unary).
//
//   nvcc -cubin -arch=sm_103a -o kernels/dflash_select.cubin tools/kernels-src/dflash_select.cu
#include <cuda_bf16.h>

#define KC 16

extern "C" __global__ void kern_dflash_select(
    const long long* __restrict__ cand, const float* __restrict__ unary,
    const __nv_bfloat16* __restrict__ hidden_r, const __nv_bfloat16* __restrict__ succ,
    const __nv_bfloat16* __restrict__ pred, const __nv_bfloat16* __restrict__ pred_anchor,
    long long* __restrict__ out, int steps, int rank, int hr_stride) {
  extern __shared__ float red[];   // [KC][rank]
  __shared__ float score[KC];
  __shared__ int prev;
  const int r = threadIdx.x;
  if (r == 0) prev = 0;
  __syncthreads();
  for (int l = 0; l < steps; ++l) {
    const __nv_bfloat16* pr = l == 0 ? pred_anchor : pred + ((long long)(l - 1) * KC + prev) * rank;
    const float ph = __bfloat162float(__float2bfloat16(
        __bfloat162float(pr[r]) * __bfloat162float(hidden_r[(long long)l * hr_stride + r])));
    for (int c = 0; c < KC; ++c)
      red[c * rank + r] = ph * __bfloat162float(succ[((long long)l * KC + c) * rank + r]);
    __syncthreads();
    for (int stride = rank / 2; stride > 0; stride >>= 1) {
      if (r < stride)
        for (int c = 0; c < KC; ++c) red[c * rank + r] += red[c * rank + r + stride];
      __syncthreads();
    }
    if (r < KC) score[r] = unary[l * KC + r] + __bfloat162float(__float2bfloat16(red[r * rank]));
    __syncthreads();
    if (r == 0) {
      int best = 0;
      for (int c = 1; c < KC; ++c) if (score[c] > score[best]) best = c;
      out[l] = cand[l * KC + best];
      prev = best;
    }
    __syncthreads();
  }
}

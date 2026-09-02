// Kimi-K3 absorbed-MLA decode over the paged latent KV cache — pegainfer's
// certified `csrc/k3/k3_mla_paged_attn.cu` kernel (pegainfer-kernels), taken
// verbatim as a plain extern "C" entry so kern can launch it from a manifest.
//
//   kern_k3_mla_paged_attn(q, w_kv_b, cache, table, max_pages, page_stride,
//                          n, scale, o)      grid (b, heads), block 128
//
// `cache` is the pool slab shifted to this layer's slice (elements); a page
// holds 64 tokens of 576-wide bf16 latent rows (post-norm kv latent | rope
// half — NoPE); `table[b, max_pages]` maps logical page -> physical page
// (the walk covers `n[b]` tokens, so entries past the context are never
// read); `page_stride` is the page-to-page distance in elements. Per head:
// absorb q_nope through W_UK, score the shared rows with bf16 landing, one
// online softmax over 64-token pages in f32, expand the attended latent with
// W_UV. See pegainfer's source for the documented rounding chain; nothing in
// the arithmetic changed here.
//
//   nvcc -cubin -arch=sm_103a -o kernels/k3_mla_paged_attn.cubin tools/kernels-src/k3_mla_paged_attn.cu
#include <cuda_bf16.h>

#define WARP_SIZE 32

__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
  }
  return val;
}

namespace {
constexpr int kNope = 128;    // qk_nope_head_dim
constexpr int kRope = 64;     // qk_rope_head_dim
constexpr int kLatent = 512;  // kv_lora_rank
constexpr int kRow = kLatent + kRope;  // cached latent row width
constexpr int kVd = 128;      // v_head_dim
constexpr int kPageTokens = 64;
constexpr int kThreads = 128;
constexpr int kWarps = kThreads / WARP_SIZE;
constexpr int kDimsPerThread = kLatent / kThreads;
constexpr float kNeg = -1.0e30f;

// Fixed-order block reductions: warp shuffle trees, then thread 0 folds the
// warp partials in ascending warp order. The result is broadcast via stage[0].
__device__ __forceinline__ float block_max(float value, float* stage) {
  value = warp_reduce_max(value);
  if ((threadIdx.x & (WARP_SIZE - 1)) == 0) stage[threadIdx.x / WARP_SIZE] = value;
  __syncthreads();
  if (threadIdx.x == 0) {
    float folded = stage[0];
    for (int w = 1; w < kWarps; ++w) folded = fmaxf(folded, stage[w]);
    stage[0] = folded;
  }
  __syncthreads();
  float out = stage[0];
  __syncthreads();
  return out;
}

__device__ __forceinline__ float block_sum(float value, float* stage) {
  value = warp_reduce_sum(value);
  if ((threadIdx.x & (WARP_SIZE - 1)) == 0) stage[threadIdx.x / WARP_SIZE] = value;
  __syncthreads();
  if (threadIdx.x == 0) {
    float folded = stage[0];
    for (int w = 1; w < kWarps; ++w) folded += stage[w];
    stage[0] = folded;
  }
  __syncthreads();
  float out = stage[0];
  __syncthreads();
  return out;
}

}  // namespace

extern "C" __global__ void kern_k3_mla_paged_attn(
    const __nv_bfloat16* __restrict__ q,        // [b, heads * 192]
    const __nv_bfloat16* __restrict__ w_kv_b,   // [heads * 256, 512]
    const __nv_bfloat16* __restrict__ cache,    // layer-shifted slab base
    const int* __restrict__ table,              // [b, max_pages]
    int max_pages,
    long long page_stride,                      // elements from page to page
    const int* __restrict__ n,                  // [b] context lengths
    const __nv_bfloat16* __restrict__ scale,    // [1] softmax scale
    __nv_bfloat16* __restrict__ o) {            // [b, heads * 128]
  const int bb = blockIdx.x;
  const int bh = blockIdx.y;
  const int heads = gridDim.y;
  const int tid = threadIdx.x;
  const int warp = tid / WARP_SIZE;
  const int lane = tid & (WARP_SIZE - 1);
  const int ctx = n[bb];
  const __nv_bfloat16* qh =
      q + ((size_t)bb * heads + bh) * (size_t)(kNope + kRope);
  const int* bt = table + (size_t)bb * max_pages;
  const __nv_bfloat16 sc = scale[0];

  __shared__ alignas(8) __nv_bfloat16 q_abs[kRow];
  __shared__ float scores[kPageTokens];
  __shared__ float stage[kWarps];

  // The absorbed query: q_abs = [W_UK[h]^T q_nope | q_rope], f32 accumulate,
  // one bf16 landing (mirroring every projection's landing discipline).
  const __nv_bfloat16* w_uk = w_kv_b + (size_t)bh * 2 * kVd * kLatent;
  for (int j = tid; j < kLatent; j += kThreads) {
    float acc = 0.0f;
    for (int d = 0; d < kNope; ++d) {
      acc += __bfloat162float(qh[d]) * __bfloat162float(w_uk[(size_t)d * kLatent + j]);
    }
    q_abs[j] = __float2bfloat16_rn(acc);
  }
  for (int j = tid; j < kRope; j += kThreads) {
    q_abs[kLatent + j] = qh[kNope + j];
  }
  __syncthreads();

  const __nv_bfloat162* q2 = reinterpret_cast<const __nv_bfloat162*>(q_abs);
  const int chunks = (ctx + kPageTokens - 1) / kPageTokens;

  // A page of landed scores: a warp per token (warps stride the page), lanes
  // stride the 576 dims in bf16 pairs, one fixed-order shuffle tree per token,
  // then the retired kernel's landing — bf16(dot) scaled in bf16. The landing
  // also absorbs the shuffle tree's f32 summation-order noise, and it makes
  // the recompute in the attend pass bit-identical to the stats pass.
  auto page_scores = [&](const __nv_bfloat16* cpage, int len) {
    for (int t = warp; t < len; t += kWarps) {
      float acc = 0.0f;
      if (cpage != nullptr) {
        const __nv_bfloat162* c2 = reinterpret_cast<const __nv_bfloat162*>(
            cpage + (size_t)t * kRow);
        for (int d = lane; d < kRow / 2; d += WARP_SIZE) {
          const float2 cf = __bfloat1622float2(c2[d]);
          const float2 qf = __bfloat1622float2(q2[d]);
          acc += qf.x * cf.x + qf.y * cf.y;
        }
        acc = warp_reduce_sum(acc);
      }
      if (lane == 0) {
        scores[t] = __bfloat162float(__hmul(__float2bfloat16_rn(acc), sc));
      }
    }
  };

  // Stats pass: the running score maximum and, online against it, the
  // softmax denominator. The max over landed scores is order-free, so it is
  // exactly the retired kernel's; the denominator differs from a flat
  // ascending-t sum only in f32 summation order — noise the bf16 prob
  // landing below absorbs.
  float m_run = kNeg;
  float l_run = 0.0f;
  for (int chunk = 0; chunk < chunks; ++chunk) {
    const int base = chunk * kPageTokens;
    const int len = min(kPageTokens, ctx - base);
    const int page = bt[chunk];
    const __nv_bfloat16* cpage =
        page >= 0 ? cache + (long long)page * page_stride : nullptr;
    __syncthreads();  // scores consumed by the previous chunk's reductions
    page_scores(cpage, len);
    __syncthreads();
    float local = kNeg;
    for (int t = tid; t < len; t += kThreads) local = fmaxf(local, scores[t]);
    const float page_max = block_max(local, stage);
    const float m_new = fmaxf(m_run, page_max);
    local = 0.0f;
    for (int t = tid; t < len; t += kThreads) {
      local += expf(scores[t] - m_new);
    }
    const float page_sum = block_sum(local, stage);
    l_run = l_run * expf(m_run - m_new) + page_sum;  // exp(-1e30)==0 on entry
    m_run = m_new;
  }

  // Attend pass: recompute each page's landed scores (bit-identical by
  // construction), take the retired kernel's bf16 probabilities against the
  // final max/denominator, and accumulate the latent row — threads own
  // kDimsPerThread strided latent dims, tokens ascending.
  float oacc[kDimsPerThread];
  for (int i = 0; i < kDimsPerThread; ++i) oacc[i] = 0.0f;
  for (int chunk = 0; chunk < chunks; ++chunk) {
    const int base = chunk * kPageTokens;
    const int len = min(kPageTokens, ctx - base);
    const int page = bt[chunk];
    const __nv_bfloat16* cpage =
        page >= 0 ? cache + (long long)page * page_stride : nullptr;
    __syncthreads();  // scores consumed by the previous chunk's attend
    page_scores(cpage, len);
    __syncthreads();
    for (int t = tid; t < len; t += kThreads) {
      scores[t] = __bfloat162float(
          __float2bfloat16_rn(expf(scores[t] - m_run) / l_run));
    }
    __syncthreads();
    if (cpage != nullptr) {
      for (int i = 0; i < kDimsPerThread; ++i) {
        const int j = i * kThreads + tid;
        float acc = oacc[i];
        for (int t = 0; t < len; ++t) {
          acc += scores[t] * __bfloat162float(cpage[(size_t)t * kRow + j]);
        }
        oacc[i] = acc;
      }
    }
  }

  // Land the attended latent row in bf16.
  __shared__ __nv_bfloat16 o_lat[kLatent];
  for (int i = 0; i < kDimsPerThread; ++i) {
    o_lat[i * kThreads + tid] = __float2bfloat16_rn(oacc[i]);
  }
  __syncthreads();

  // The W_UV expansion, one value dim per thread.
  const __nv_bfloat16* w_uv = w_kv_b + ((size_t)bh * 2 * kVd + kNope) * kLatent;
  const int dv = tid;  // kThreads == kVd
  float acc = 0.0f;
  for (int j = 0; j < kLatent; ++j) {
    acc += __bfloat162float(w_uv[(size_t)dv * kLatent + j]) *
           __bfloat162float(o_lat[j]);
  }
  o[((size_t)bb * heads + bh) * (size_t)kVd + dv] = __float2bfloat16_rn(acc);
}


// Kimi-K3 absorbed-MLA decode over the paged latent KV cache — head-grouped,
// split-KV rewrite of pegainfer's one-block-per-head kernel.
//
//   extern "C" __global__ void kern_k3_mla_paged_attn(
//       const f32*  q_partial,   // [B, HEADS*192]  f32 partial, nope 128 | rope 64
//       const bf16* w_kv_b,      // [HEADS*256, 512]  W_UK = rows h*256+0..128,
//                                //                   W_UV = rows h*256+128..256
//       const bf16* cache,       // slab base already shifted to this layer
//       const int*  block_table, // [B, max_pages]
//       int max_pages, long long page_stride,   // page p at p*page_stride elements
//       const int*  seq_lens,    // [B]  includes the current token
//       const bf16* scale,       // [1]  192^-0.5
//       const bf16* mla_gate,    // [B, HEADS*128]
//       bf16*       gated,       // [B, HEADS*128]
//       int B);
//
//   grid    (B, 48, 1)      gridDim.y = HEAD_GROUPS = 6 head groups * 8 KV splits
//   block   (512, 1, 1)
//   cluster (1, 8, 1)       the 8 splits of one head group form one cluster
//   smem    0 dynamic. The 216320 B of state is STATIC __shared__: ptxas and the
//           driver take the full carveout on sm_103a without the >48 KB
//           cuFuncSetAttribute opt-in, so launch it with sharedMemBytes = 0.
//
// blockIdx.y = g*8 + s: `g` picks 16 of the 96 heads, `s` picks one eighth of the
// page list. A block reads its page slice ONCE for all 16 of its heads (every head
// shares the latent rows — MQA-like), so a layer's KV traffic is 6 group-reads
// instead of the old kernel's 96 head-reads. Both products are mma.sync bf16
// tensor-core ops over a cp.async-staged, bank-skewed page:
//   scores  [16 heads x 576] x [576 x 64 tokens]   m16n8k16, q_abs pinned in regs
//   latent  [16 heads x 64 tokens] x [64 x 512]    m16n8k16, V via ldmatrix.trans
// The eight splits of a cluster exchange q_abs, (m, l) and the latent partials
// through distributed shared memory, so there is no global scratch and one launch.
// No multicast TMA anywhere (plain cp.async + ldmatrix only).
//
// Landing points (docs/k3-kernel-abi.md K5) — pegainfer's chain, unchanged:
//   q_h     = bf16(q_partial[b, h*192 .. +192])
//   q_abs   = [ bf16(sum_d q_h[d]*W_UK_h[d,j]) for j<512 | q_h[128..192] ]   f32 acc
//   s[t]    = f32( bf16(dot(q_abs, row_t)) * scale )   bf16 landing, bf16 multiply
//   m,l     = online softmax in f32 over 64-token pages, reduced across the splits
//   p[t]    = f32(bf16(exp(s[t]-m)/l))                 against the FINAL m and l,
//                                                      hence the second page walk
//   lat[j]  = bf16(sum_t p[t]*row_t[j])                f32 acc
//   o[dv]   = bf16(sum_j W_UV_h[dv,j]*lat[j])          f32 acc
//   gated   = o[dv] * bf16(sigmoid(f32(mla_gate[...])))
//
//   nvcc -cubin -arch=sm_103a -O3 tools/kernels-src/k3_mla_paged_attn.cu
#include <cuda_bf16.h>

#define K5_HEADS   96
#define K5_GRP     16                      // heads per block group
#define K5_NGRP    (K5_HEADS / K5_GRP)     // 6 head groups
#ifndef K5_SPL
#define K5_SPL     8                     // KV splits per group == cluster size
#endif
#define K5_GROUPS  (K5_NGRP * K5_SPL)      // 48 == gridDim.y
#ifndef K5_THREADS
#define K5_THREADS 512
#endif
#define K5_WARPS   (K5_THREADS / 32)       // 16
#define K5_NOPE    128
#define K5_ROPE    64
#define K5_LAT     512
#define K5_ROW     576                     // latent 512 | rope 64
#define K5_PAGE    64
#define K5_PAD     584                     // page/q_abs row pitch: 1168 B, 16 B
                                           // aligned, 292 words -> 4 banks of skew
#define K5_PPAD    72                      // probability-matrix row pitch
#define K5_SCPAD   68                      // f32 score row pitch
#define K5_APAD    516                     // latent-partial row pitch (bank skew)
#define K5_HPB     (K5_GRP / K5_SPL)       // heads finalized per block = 2
#define K5_NEG     (-1.0e30f)

// Score-product warp tiling: NW token groups x KW k groups.
#define K5_NW      8                       // token groups (8 tokens each)
#define K5_KW      (K5_WARPS / K5_NW)      // 2 k groups
#define K5_KSPAN   (K5_ROW / K5_KW)        // 288 dims per warp
#define K5_KGRP    (K5_KSPAN / 32)         // 9 ldmatrix.x4 loads per page
#define K5_PVNT    (K5_LAT / 8 / K5_WARPS) // 4 latent n-tiles per warp
#define K5_RCH     (K5_ROW / 8)            // 72 16-byte chunks per cached row
#define K5_CHIT    (K5_PAGE * K5_RCH / K5_THREADS)  // chunks staged per thread
#define K5_CHR     (K5_THREADS / K5_RCH)   // row advance per chunk step
#define K5_CHO     (K5_THREADS % K5_RCH)   // column advance per chunk step

struct K5Smem {
  __nv_bfloat16 pg[2][K5_PAGE][K5_PAD];     // 149504  staged pages, double buffered
  __nv_bfloat16 q_abs[K5_GRP][K5_PAD];      //  18688  absorbed queries, group-wide
  __nv_bfloat16 pmat[K5_GRP][K5_PPAD];      //   2304  landed probabilities
  float         sc2[K5_KW][K5_GRP][K5_SCPAD];  // 8704  raw score partials, per k group
  float         ml[K5_GRP][2];              //    128  this split's running (m, l)
  float         mlg[K5_GRP][2];             //    128  cluster-reduced (m, l)
  float         accx[K5_GRP][K5_APAD];      //  33024  this split's latent partial
  __nv_bfloat16 lat[K5_HPB][K5_LAT];        //   2048  merged latent, landed
  __nv_bfloat16 qh[K5_HPB][192];            //    768  landed q for the owned heads
  float         ovec[K5_HPB][128];          //   1024  o before the gate, for a
                                            //         coalesced epilogue store
};
#define K5_SMEM ((int)sizeof(K5Smem))

// ---------------------------------------------------------------- PTX helpers
__device__ __forceinline__ unsigned smem_u32(const void* p) {
  return (unsigned)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void ldm_x4(unsigned (&r)[4], unsigned a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void ldm_x4_t(unsigned (&r)[4], unsigned a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void mma_16816(float (&d)[4], const unsigned (&a)[4],
                                          unsigned b0, unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void cp_async16(unsigned dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;"); }
template <int N>
__device__ __forceinline__ void cp_wait() {
  asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory");
}
__device__ __forceinline__ unsigned cta_rank() {
  unsigned r; asm("mov.u32 %0, %%cluster_ctarank;" : "=r"(r)); return r;
}
__device__ __forceinline__ unsigned map_rank(unsigned a, unsigned r) {
  unsigned o; asm("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(o) : "r"(a), "r"(r)); return o;
}
__device__ __forceinline__ void cluster_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
  asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
}
__device__ __forceinline__ float ldc_f32(unsigned a) {
  float v; asm volatile("ld.shared::cluster.f32 %0, [%1];" : "=f"(v) : "r"(a)); return v;
}
__device__ __forceinline__ uint4 ldc_v4(unsigned a) {
  uint4 v;
  asm volatile("ld.shared::cluster.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a));
  return v;
}

// ----------------------------------------------------------------- the kernel
extern "C" __global__ void __launch_bounds__(K5_THREADS, 1) __cluster_dims__(1, K5_SPL, 1)
kern_k3_mla_paged_attn(const float* __restrict__ q_partial,
                       const __nv_bfloat16* __restrict__ w_kv_b,
                       const __nv_bfloat16* __restrict__ cache,
                       const int* __restrict__ block_table, int max_pages,
                       long long page_stride, const int* __restrict__ seq_lens,
                       const __nv_bfloat16* __restrict__ scale,
                       const __nv_bfloat16* __restrict__ mla_gate,
                       __nv_bfloat16* __restrict__ gated, int B) {
  // Static shared memory: ptxas and the driver accept the full 214 KB carveout
  // on sm_103a without the >48 KB dynamic opt-in, so the launch needs no
  // cuFuncSetAttribute and `shared_mem` in the manifest stays 0.
  __shared__ __align__(16) K5Smem sm;

  const int b = blockIdx.x;
  const int g = blockIdx.y / K5_SPL;
  const int s = (int)cta_rank();                 // == blockIdx.y % K5_SPL
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int head0 = g * K5_GRP;
  const int ctx = seq_lens[b];
  const int npages = (ctx + K5_PAGE - 1) / K5_PAGE;
  const int per = (npages + K5_SPL - 1) / K5_SPL;
  int p_lo = s * per; if (p_lo > npages) p_lo = npages;
  int p_hi = p_lo + per; if (p_hi > npages) p_hi = npages;
  const int* __restrict__ bt = block_table + (long long)b * max_pages;
  const __nv_bfloat16 sc_bf = scale[0];

  // ---- q_abs for the heads this split owns, then all-gather over the cluster
  for (int i = tid; i < K5_HPB * 192; i += K5_THREADS) {
    const int u = i / 192, k = i - u * 192;
    const int gh = head0 + s * K5_HPB + u;
    sm.qh[u][k] =
        __float2bfloat16_rn(q_partial[(size_t)b * (K5_HEADS * 192) + (size_t)gh * 192 + k]);
  }
  if (tid < K5_GRP) { sm.ml[tid][0] = K5_NEG; sm.ml[tid][1] = 0.f; }
  __syncthreads();
  for (int i = tid; i < K5_HPB * (K5_LAT / 2); i += K5_THREADS) {
    const int u = i / (K5_LAT / 2), j2 = i - u * (K5_LAT / 2);
    const __nv_bfloat162* __restrict__ w2 = reinterpret_cast<const __nv_bfloat162*>(
        w_kv_b + (size_t)(head0 + s * K5_HPB + u) * 256 * K5_LAT);
    // four independent accumulator chains, 16 loads in flight: this matvec is
    // pure global-load latency (2 x 128 KB of W_UK per block) and nothing else
    // in the kernel overlaps it.
    float ax[4] = {0.f, 0.f, 0.f, 0.f}, ay[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 4
    for (int d = 0; d < K5_NOPE; d += 4) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float q = __bfloat162float(sm.qh[u][d + e]);
        const float2 w = __bfloat1622float2(w2[(size_t)(d + e) * (K5_LAT / 2) + j2]);
        ax[e] += q * w.x;
        ay[e] += q * w.y;
      }
    }
    sm.q_abs[s * K5_HPB + u][2 * j2] =
        __float2bfloat16_rn((ax[0] + ax[1]) + (ax[2] + ax[3]));
    sm.q_abs[s * K5_HPB + u][2 * j2 + 1] =
        __float2bfloat16_rn((ay[0] + ay[1]) + (ay[2] + ay[3]));
  }
  for (int i = tid; i < K5_HPB * (K5_PAD - K5_LAT); i += K5_THREADS) {
    const int u = i / (K5_PAD - K5_LAT), r = i - u * (K5_PAD - K5_LAT);
    sm.q_abs[s * K5_HPB + u][K5_LAT + r] =
        (r < K5_ROPE) ? sm.qh[u][K5_NOPE + r] : __float2bfloat16_rn(0.f);
  }
  __syncthreads();
  cluster_sync();
  {
    const unsigned my = smem_u32(&sm.q_abs[0][0]);
    uint4* dst = reinterpret_cast<uint4*>(&sm.q_abs[0][0]);
#pragma unroll
    for (int p = 0; p < K5_SPL; ++p) {
      if (p == s) continue;
      const unsigned peer = map_rank(my, p);
      const int base = p * K5_HPB * K5_PAD / 8;
      for (int c = tid; c < K5_HPB * K5_PAD / 8; c += K5_THREADS)
        dst[base + c] = ldc_v4(peer + (unsigned)((base + c) * 16));
    }
  }
  __syncthreads();

  // ---------------------------------------------------------------- page walk
  const int kwi = warp / K5_NW;                 // k group: dims [kwi*KSPAN, +KSPAN)
  const int nwi = warp % K5_NW;                 // token group: 8 tokens
  const int dim0 = warp * (8 * K5_PVNT);        // this warp's latent columns

  // q_abs is the A operand of the score product and never changes: pin the
  // 16 x KSPAN slab for this warp's k group in registers, once.
  unsigned qf[2 * K5_KGRP][4];
  {
    const unsigned qa = smem_u32(&sm.q_abs[lane & 15][kwi * K5_KSPAN + 8 * (lane >> 4)]);
#pragma unroll
    for (int i = 0; i < 2 * K5_KGRP; ++i) ldm_x4(qf[i], qa + (unsigned)(i * 32));
  }

  auto issue = [&](int page, int buf, int len) {
    if (page < 0 || len <= 0) { cp_commit(); return; }
    const __nv_bfloat16* src = cache + (long long)page * page_stride;
    int row = tid / K5_RCH, off = tid - row * K5_RCH;
    unsigned d = smem_u32(&sm.pg[buf][0][0]) + (unsigned)((row * K5_PAD + off * 8) * 2);
    const __nv_bfloat16* sp = src + (size_t)tid * 8;
#pragma unroll
    for (int i = 0; i < K5_CHIT; ++i) {
      if (row < len) cp_async16(d, sp);
      sp += K5_THREADS * 8;
      off += K5_CHO;
      if (off >= K5_RCH) {
        off -= K5_RCH;
        row += K5_CHR + 1;
        d += (unsigned)(((K5_CHR + 1) * K5_PAD - (K5_RCH - K5_CHO) * 8) * 2);
      } else {
        row += K5_CHR;
        d += (unsigned)((K5_CHR * K5_PAD + K5_CHO * 8) * 2);
      }
    }
    cp_commit();
  };
  auto zero_tail = [&](int buf, int len) {
    const uint4 z = make_uint4(0, 0, 0, 0);
    uint4* p = reinterpret_cast<uint4*>(&sm.pg[buf][0][0]);
    const int lo = len < 0 ? 0 : len;
    for (int c = tid; c < (K5_PAGE - lo) * (K5_PAD / 8); c += K5_THREADS) {
      const int row = c / (K5_PAD / 8), off = c - row * (K5_PAD / 8);
      p[(lo + row) * (K5_PAD / 8) + off] = z;
    }
  };
  auto page_len = [&](int pi) {
    const int l = ctx - pi * K5_PAGE;
    return l > K5_PAGE ? K5_PAGE : l;
  };

  // raw f32 score partials for the staged page -> sm.sc2[kwi][h][t]; the two k
  // groups are folded and landed by whoever consumes them (no barrier in here).
  auto scores = [&](int buf) {
    float c[4] = {0.f, 0.f, 0.f, 0.f};
    unsigned bf[4];
    const unsigned pb =
        smem_u32(&sm.pg[buf][nwi * 8 + (lane & 7)][kwi * K5_KSPAN + 8 * (lane >> 3)]);
#pragma unroll
    for (int kk = 0; kk < K5_KGRP; ++kk) {
      ldm_x4(bf, pb + (unsigned)(kk * 64));
      mma_16816(c, qf[2 * kk], bf[0], bf[1]);
      mma_16816(c, qf[2 * kk + 1], bf[2], bf[3]);
    }
    const int hA = lane >> 2, t0 = nwi * 8 + 2 * (lane & 3);
#pragma unroll
    for (int i = 0; i < 4; ++i)
      sm.sc2[kwi][hA + ((i >> 1) << 3)][t0 + (i & 1)] = c[i];
  };
  // fold the k groups and apply the documented landing
  auto land = [&](int h, int t, int len) {
    const float v = sm.sc2[0][h][t] + sm.sc2[1][h][t];
    return (t < len) ? __bfloat162float(__hmul(__float2bfloat16_rn(v), sc_bf)) : K5_NEG;
  };

  // ---- pass 1: running (m, l) over this split's pages
  {
    const int n = p_hi - p_lo;
    int ea = (n > 0) ? bt[p_lo] : -1, eb = (n > 1) ? bt[p_lo + 1] : -1;
    if (n > 0) issue(ea, 0, page_len(p_lo));
    for (int i = 0; i < n; ++i) {
      const int buf = i & 1, len = page_len(p_lo + i);
      const int ec = (i + 2 < n) ? bt[p_lo + i + 2] : -1;
      if (i + 1 < n) { issue(eb, buf ^ 1, page_len(p_lo + i + 1)); cp_wait<1>(); }
      else cp_wait<0>();
      __syncthreads();
      if (len < K5_PAGE || ea < 0) {
        zero_tail(buf, ea < 0 ? 0 : len);
        __syncthreads();
      }
      scores(buf);
      __syncthreads();
      if (warp < K5_GRP) {
        const int h = warp;
        const float v0 = land(h, lane, len), v1 = land(h, lane + 32, len);
        float mx = fmaxf(v0, v1);
#pragma unroll
        for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
        const float m_old = sm.ml[h][0];
        const float m_new = fmaxf(m_old, mx);
        float sum = expf(v0 - m_new) + expf(v1 - m_new);
#pragma unroll
        for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
        if (lane == 0) {
          const float rs = (m_old <= K5_NEG) ? 0.f : expf(m_old - m_new);
          sm.ml[h][0] = m_new;
          sm.ml[h][1] = sm.ml[h][1] * rs + sum;
        }
      }
      ea = eb; eb = ec;
      __syncthreads();
    }
  }

  // ---- cluster reduce (m, l)
  cluster_sync();
  if (tid < K5_GRP) {
    const unsigned my = smem_u32(&sm.ml[tid][0]);
    float ms[K5_SPL], ls[K5_SPL], mg = K5_NEG;
#pragma unroll
    for (int p = 0; p < K5_SPL; ++p) {
      const unsigned a = map_rank(my, p);
      ms[p] = ldc_f32(a);
      ls[p] = ldc_f32(a + 4u);
      mg = fmaxf(mg, ms[p]);
    }
    float lg = 0.f;
#pragma unroll
    for (int p = 0; p < K5_SPL; ++p)
      lg += (ms[p] <= K5_NEG) ? 0.f : ls[p] * expf(ms[p] - mg);
    sm.mlg[tid][0] = mg;
    sm.mlg[tid][1] = lg;
  }
  __syncthreads();

  // ---- pass 2: p against the final (m, l), latent accumulation
  float acc[K5_PVNT][4];
#pragma unroll
  for (int i = 0; i < K5_PVNT; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[i][j] = 0.f;
  {
    const int n = p_hi - p_lo;
    int ea = (n > 0) ? bt[p_lo] : -1, eb = (n > 1) ? bt[p_lo + 1] : -1;
    if (n > 0) issue(ea, 0, page_len(p_lo));
    for (int i = 0; i < n; ++i) {
      const int buf = i & 1, len = page_len(p_lo + i);
      const int ec = (i + 2 < n) ? bt[p_lo + i + 2] : -1;
      if (i + 1 < n) { issue(eb, buf ^ 1, page_len(p_lo + i + 1)); cp_wait<1>(); }
      else cp_wait<0>();
      __syncthreads();
      if (len < K5_PAGE || ea < 0) {
        zero_tail(buf, ea < 0 ? 0 : len);
        __syncthreads();
      }
      scores(buf);
      __syncthreads();
      for (int idx = tid; idx < K5_GRP * K5_PAGE; idx += K5_THREADS) {
        const int h = idx >> 6, t = idx & 63;
        const float sv = land(h, t, len);
        sm.pmat[h][t] = (sv > K5_NEG)
            ? __float2bfloat16_rn(expf(sv - sm.mlg[h][0]) / sm.mlg[h][1])
            : __float2bfloat16_rn(0.f);
      }
      __syncthreads();
      {
        unsigned a[4], bf[4];
        const unsigned pa = smem_u32(&sm.pmat[lane & 15][8 * (lane >> 4)]);
        const unsigned vb = smem_u32(
            &sm.pg[buf][(lane & 7) + 8 * ((lane >> 3) & 1)][dim0 + 8 * (lane >> 4)]);
#pragma unroll
        for (int kt = 0; kt < 4; ++kt) {
          ldm_x4(a, pa + (unsigned)(kt * 32));
#pragma unroll
          for (int nt = 0; nt < K5_PVNT; nt += 2) {
            ldm_x4_t(bf, vb + (unsigned)((kt * 16 * K5_PAD + nt * 8) * 2));
            mma_16816(acc[nt], a, bf[0], bf[1]);
            mma_16816(acc[nt + 1], a, bf[2], bf[3]);
          }
        }
      }
      ea = eb; eb = ec;
      __syncthreads();
    }
  }

  // ---- publish this split's latent partial, merge over the cluster
#pragma unroll
  for (int nt = 0; nt < K5_PVNT; ++nt) {
    const int d = dim0 + nt * 8 + 2 * (lane & 3);
    sm.accx[lane >> 2][d] = acc[nt][0];
    sm.accx[lane >> 2][d + 1] = acc[nt][1];
    sm.accx[(lane >> 2) + 8][d] = acc[nt][2];
    sm.accx[(lane >> 2) + 8][d + 1] = acc[nt][3];
  }
  __syncthreads();
  cluster_sync();
#pragma unroll
  for (int u = 0; u < K5_HPB; ++u) {
    const int h = s * K5_HPB + u;
    const unsigned my = smem_u32(&sm.accx[h][0]);
    float v[K5_SPL];
#pragma unroll
    for (int p = 0; p < K5_SPL; ++p)
      v[p] = ldc_f32(map_rank(my, p) + (unsigned)(tid * sizeof(float)));
    float t0 = 0.f;
#pragma unroll
    for (int p = 0; p < K5_SPL; ++p) t0 += v[p];      // ascending split == ascending t
    sm.lat[u][tid] = __float2bfloat16_rn(t0);
  }
  __syncthreads();

  // ---- W_UV expansion + sigmoid gate for the heads this split owns
#pragma unroll
  for (int u = 0; u < K5_HPB; ++u) {
    const int gh = head0 + s * K5_HPB + u;
    const __nv_bfloat16* __restrict__ w_uv = w_kv_b + ((size_t)gh * 256 + K5_NOPE) * K5_LAT;
    const __nv_bfloat162* lat2 = reinterpret_cast<const __nv_bfloat162*>(&sm.lat[u][0]);
#pragma unroll 2
    for (int dv = warp; dv < 128; dv += K5_WARPS) {
      const __nv_bfloat162* w2 =
          reinterpret_cast<const __nv_bfloat162*>(w_uv + (size_t)dv * K5_LAT);
      float a = 0.f;
#pragma unroll
      for (int it = 0; it < 8; ++it) {
        const int j2 = lane + it * 32;
        const float2 wf = __bfloat1622float2(w2[j2]);
        const float2 lf = __bfloat1622float2(lat2[j2]);
        a += wf.x * lf.x + wf.y * lf.y;
      }
#pragma unroll
      for (int o = 16; o; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
      if (lane == 0) sm.ovec[u][dv] = a;
    }
  }
  // One 2-byte store per warp would burn a 32-byte sector each; stage o in
  // shared memory and let consecutive threads own consecutive dv instead.
  __syncthreads();
  for (int i = tid; i < K5_HPB * 128; i += K5_THREADS) {
    const int u = i >> 7, dv = i & 127;
    const size_t oi =
        (size_t)b * (K5_HEADS * 128) + (size_t)(head0 + s * K5_HPB + u) * 128 + dv;
    const float gf = __bfloat162float(mla_gate[oi]);
    gated[oi] = __hmul(__float2bfloat16_rn(sm.ovec[u][dv]),
                       __float2bfloat16_rn(1.0f / (1.0f + expf(-gf))));
  }
}

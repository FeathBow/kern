// kern's AOT MegaMoE kernels for pruned K3 (224 routed experts), one entry
// per world. Built by tools/build_k3_mega.sh into k3_mega_moe.cubin.
//
// Launch ABI (kern manifest params, in order):
//   out buffer<bf16> y            [tokens, 3584]
//   out buffer<i32>  stats        [experts/ranks] cumulative recv counts (red.add)
//   i32              tokens       this rank's token count (<= 16896)
//   in  buffer<u64>  peer_bases   the `ep` peer array: every rank's slab base
//   i32              rank         this rank's index in `ep`
//   tensormap x 18                l1_acts, l1_acts_sf, l1_weights,
//                                 l1_weights_sf, l1_output, l2_acts,
//                                 l2_acts_sf, l2_weights, l2_weights_sf, then
//                                 the same nine again for the (absent) shared
//                                 expert — upstream aliases them.
// block [kNumThreads,1,1], grid [152,1,1], cluster [2,1,1], shared_mem kSmemSize
// (see layout_dump for the numbers per world).
#include "k3_mega_config.cuh"

#ifndef K3_MEGA_INF_DEFINED
#define K3_MEGA_INF_DEFINED
constexpr float inf = __builtin_huge_valf();
#endif

#include "sm100_fp8_fp4_mega_moe.cuh"

namespace {

template <int kExperts, int kRanks>
struct World {
  static constexpr int kCfg = k3_mega::pinned_config<kExperts, kRanks>();
  using G = k3_mega::MegaGeom<kExperts, kRanks, kCfg>;
  using Ring = k3_mega::MegaRing<kExperts, kRanks>;

  template <bool kSitu>
  CUTLASS_DEVICE static void run(void* y, int* stats, uint32_t num_tokens,
                                 const int64_t* peer_bases, uint32_t rank_idx,
                                 const cute::TmaDescriptor& t0, const cute::TmaDescriptor& t1,
                                 const cute::TmaDescriptor& t2, const cute::TmaDescriptor& t3,
                                 const cute::TmaDescriptor& t4, const cute::TmaDescriptor& t5,
                                 const cute::TmaDescriptor& t6, const cute::TmaDescriptor& t7,
                                 const cute::TmaDescriptor& t8, const cute::TmaDescriptor& s0,
                                 const cute::TmaDescriptor& s1, const cute::TmaDescriptor& s2,
                                 const cute::TmaDescriptor& s3, const cute::TmaDescriptor& s4,
                                 const cute::TmaDescriptor& s5, const cute::TmaDescriptor& s6,
                                 const cute::TmaDescriptor& s7, const cute::TmaDescriptor& s8) {
    deep_gemm::sm100_fp8_fp4_mega_moe_body<
        k3_mega::kMaxTokensPerRank, k3_mega::kHidden, k3_mega::kIntermediate, kExperts,
        /*shared=*/0, k3_mega::kNumTopk, G::kBlockM, k3_mega::kMegaBlockN, G::kBlockK,
        G::kStoreBlockM, G::kSfBlockM, G::kSfBlockN, Ring::kTokens, Ring::kSfTokens,
        G::kPipe.num_stages, k3_mega::kBytesPerPull, k3_mega::kNumDispatchThreads,
        k3_mega::kNumNonEpilogueThreads, G::kEpilogueThreads, k3_mega::kGb300Sms, kRanks,
        /*clamp=*/inf, kSitu, /*fast_math=*/false>(
        y, stats, num_tokens, peer_bases, rank_idx, t0, t1, t2, t3, t4, t5, t6, t7, t8, s0, s1,
        s2, s3, s4, s5, s6, s7, s8);
  }
};

}  // namespace

#define K3_MEGA_TM_PARAMS                                                                    \
  const __grid_constant__ CUtensorMap t0, const __grid_constant__ CUtensorMap t1,            \
      const __grid_constant__ CUtensorMap t2, const __grid_constant__ CUtensorMap t3,        \
      const __grid_constant__ CUtensorMap t4, const __grid_constant__ CUtensorMap t5,        \
      const __grid_constant__ CUtensorMap t6, const __grid_constant__ CUtensorMap t7,        \
      const __grid_constant__ CUtensorMap t8, const __grid_constant__ CUtensorMap s0,        \
      const __grid_constant__ CUtensorMap s1, const __grid_constant__ CUtensorMap s2,        \
      const __grid_constant__ CUtensorMap s3, const __grid_constant__ CUtensorMap s4,        \
      const __grid_constant__ CUtensorMap s5, const __grid_constant__ CUtensorMap s6,        \
      const __grid_constant__ CUtensorMap s7, const __grid_constant__ CUtensorMap s8
#define K3_MEGA_TM_ARGS t0, t1, t2, t3, t4, t5, t6, t7, t8, s0, s1, s2, s3, s4, s5, s6, s7, s8

#define K3_MEGA_WORLD(NAME, EXPERTS, RANKS, SITU)                                            \
  extern "C" __global__ __launch_bounds__(World<EXPERTS, RANKS>::G::kNumThreads, 1) void     \
  NAME(void* y, int* stats, uint32_t num_tokens, const int64_t* peer_bases, uint32_t rank,   \
       K3_MEGA_TM_PARAMS) {                                                                  \
    World<EXPERTS, RANKS>::run<SITU>(y, stats, num_tokens, peer_bases, rank, K3_MEGA_TM_ARGS); \
  }

K3_MEGA_WORLD(kern_k3_mega_moe_e224_r1_situ, 224, 1, true)
K3_MEGA_WORLD(kern_k3_mega_moe_e224_r4_situ, 224, 4, true)

// Host tool: print, as JSON, everything the manifest generator needs about a
// K3 MegaMoE world — the slab layout (`layout::MegaMoEBuffer`), the ring
// capacities, and the pinned kernel geometry.
//
//   layout_dump <experts> <ranks>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

#include "k3_mega_config.cuh"

template <int kExperts, int kRanks>
static void dump() {
  using G = k3_mega::MegaGeom<kExperts, kRanks, k3_mega::pinned_config<kExperts, kRanks>()>;
  using Ring = k3_mega::MegaRing<kExperts, kRanks>;
  const auto buffer = deep_gemm::layout::MegaMoEBuffer(
      nullptr, (uint32_t)k3_mega::kHidden, (uint32_t)k3_mega::kIntermediate, (uint32_t)kRanks,
      (uint32_t)kExperts, (uint32_t)k3_mega::kMaxTokensPerRank, (uint32_t)k3_mega::kNumTopk,
      (uint32_t)Ring::kTokens, (uint32_t)Ring::kSfTokens, /*with_sf=*/true, 0);
  auto off = [](const void* p) { return (unsigned long long)reinterpret_cast<uintptr_t>(p); };
  printf("{\n");
  printf("  \"experts\": %d, \"ranks\": %d, \"experts_per_rank\": %d,\n", kExperts, kRanks,
         kExperts / kRanks);
  printf("  \"hidden\": %d, \"intermediate\": %d, \"topk\": %d, \"max_tokens_per_rank\": %d,\n",
         k3_mega::kHidden, k3_mega::kIntermediate, k3_mega::kNumTopk,
         k3_mega::kMaxTokensPerRank);
  printf("  \"ring_tokens\": %d, \"sf_ring_tokens\": %d,\n", Ring::kTokens, Ring::kSfTokens);
  printf("  \"cfg\": %d, \"block_m\": %d, \"block_k\": %d, \"store_block_m\": %d, "
         "\"sf_block_m\": %d, \"sf_block_n\": %d, \"block_n\": %d,\n",
         k3_mega::pinned_config<kExperts, kRanks>(), G::kBlockM, G::kBlockK, G::kStoreBlockM,
         G::kSfBlockM, G::kSfBlockN, k3_mega::kMegaBlockN);
  printf("  \"num_stages\": %d, \"smem_size\": %d, \"num_threads\": %d, \"num_sms\": %d,\n",
         G::kPipe.num_stages, G::kSmemSize, G::kNumThreads, k3_mega::kGb300Sms);
  printf("  \"slab_bytes\": %llu,\n", (unsigned long long)buffer.get_num_bytes());
  printf("  \"workspace_bytes\": %llu,\n", off(buffer.workspace.get_end_ptr()));
  printf("  \"offsets\": {\n");
  printf("    \"x\": %llu, \"x_sf\": %llu, \"topk_idx\": %llu, \"topk_weights\": %llu,\n",
         off(buffer.input_token_buffer.base), off(buffer.input_sf_buffer.base),
         off(buffer.input_topk_idx_buffer.base), off(buffer.input_topk_weights_buffer.base));
  printf("    \"l1_acts\": %llu, \"l1_acts_sf\": %llu, \"l2_acts\": %llu, \"l2_acts_sf\": %llu\n",
         off(buffer.l1_token_buffer.base), off(buffer.l1_sf_buffer.base),
         off(buffer.l2_token_buffer.base), off(buffer.l2_sf_buffer.base));
  printf("  }\n}\n");
}

int main(int argc, char** argv) {
  if (argc != 3) {
    fprintf(stderr, "usage: %s <experts> <ranks>\n", argv[0]);
    return 2;
  }
  const int experts = atoi(argv[1]), ranks = atoi(argv[2]);
  if (experts == 224 && ranks == 1) dump<224, 1>();
  else if (experts == 224 && ranks == 4) dump<224, 4>();
  else {
    fprintf(stderr, "world %dx%d is not instantiated (see kern_k3_mega_moe.cu)\n", experts, ranks);
    return 1;
  }
  return 0;
}

#!/usr/bin/env python3
"""Regenerate tools/k3-mega/sm100_fp8_fp4_mega_moe.cuh from the upstream impl.

    tools/k3-mega/fork_impl.py [DEEPGEMM_ROOT]

The fork keeps the body byte-identical and only changes the signature (see
the header comment it writes). Refuses an upstream whose sha256 is not the
one the fork was reviewed against unless FORK_ANY=1.
"""
import hashlib, os, sys

REVIEWED_SHA = "e74a82f9c931ddf7"
root = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "~/pegainfer/pegainfer-kernels/third_party/DeepGEMM")
src = os.path.join(root, "deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh")
s = open(src).read()
sha = hashlib.sha256(s.encode()).hexdigest()[:16]
if sha != REVIEWED_SHA and not os.environ.get("FORK_ANY"):
    sys.exit(f"upstream sha256 {sha} != reviewed {REVIEWED_SHA}; set FORK_ANY=1 to fork anyway")

old_sig = '''CUTLASS_GLOBAL __launch_bounds__(kNumThreads, 1) void
sm100_fp8_fp4_mega_moe_impl(void* y,
                            int* cumulative_local_expert_recv_stats,
                            const uint32_t num_tokens,
                            const __grid_constant__ layout::SymBuffer<kNumRanks> sym_buffer,
'''
new_sig = '''CUTLASS_DEVICE __forceinline__ void
sm100_fp8_fp4_mega_moe_body(void* y,
                            int* cumulative_local_expert_recv_stats,
                            const uint32_t num_tokens,
                            const int64_t* kern_peer_bases,
                            const uint32_t kern_rank_idx,
'''
assert old_sig in s
s = s.replace(old_sig, new_sig)
tm = '                            const __grid_constant__ cute::TmaDescriptor tensor_map_'
assert s.count(tm) == 18, s.count(tm)
s = s.replace(tm, '                            const cute::TmaDescriptor& tensor_map_')
anchor = '    const uint32_t lane_idx = ptx::get_lane_idx();\n'
assert s.count(anchor) == 1
s = s.replace(anchor, anchor + '''
    // kern: the peer table is read on the device (see kern's sym_buffer.cuh).
    const layout::SymBuffer<kNumRanks> sym_buffer(kern_peer_bases, kern_rank_idx);
''')
hdr = f'''// kern's fork of DeepGEMM `impls/sm100_fp8_fp4_mega_moe.cuh` (pegainfer's
// vendored copy with the K3 situ activation, upstream sha256 {sha}).
// Differences from upstream, all at the signature — the body is untouched:
//   * `__global__` kernel -> `CUTLASS_DEVICE __forceinline__` body, so a
//     plain `extern "C"` wrapper (kern_k3_mega_moe.cu) can own the entry name;
//   * `SymBuffer` by value -> `(const int64_t* kern_peer_bases, uint32_t
//     kern_rank_idx)`, the view is built on the device;
//   * the 18 `__grid_constant__` tensor maps become const references to the
//     wrapper's __grid_constant__ params (same param-space addresses).
// Regenerate with tools/k3-mega/fork_impl.py.
'''
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sm100_fp8_fp4_mega_moe.cuh")
open(out, "w").write(hdr + s)
print(f"wrote {out} from upstream {sha}")

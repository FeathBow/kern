# Prebuilt kernels

Cubins built by a toolchain the repo's `nvcc` build does not run, checked in
as artifacts and pinned by sha256 like every other module. Each one has a
build recipe here; regenerate with it and commit the new bytes together.

## `mla_decode_h96_p64.cubin`

NVIDIA's Blackwell MLA decode kernel, written in the CuTe DSL and shipped as
Python source in FlashInfer (`flashinfer/cute_dsl/attention/monolithic/
mla_decode_fp16.py`, BSD-3), compiled once for K3's geometry:

| parameter          | value                                          |
|--------------------|------------------------------------------------|
| heads / q tokens   | 96 / 1                                         |
| latent / rope dims | 512 / 64, bf16                                 |
| page size          | 64 tokens                                      |
| split KV           | variable per row (`is_var_split_kv`), 256-split reducer |
| target             | sm_103a (GB300)                                |
| toolchain          | nvidia-cutlass-dsl 4.6.0, CUDA 13, FlashInfer 0.6.x |

Two entries: the split-KV attention (`kernel_cutlass_split_kv_kernel_…_0`,
384 threads, cluster 2×1×1, 232448 B dynamic smem) and its reduction
(`kernel_cutlass_reduction_kernel_…_1`, 128 threads, 1024 B smem). The
parameter ABI the manifest packs is documented in `docs/k3-kernel-abi.md`
K5; `tools/gen_k3_decode.py` writes it.

Rebuild (inside an image with FlashInfer and the CuTe DSL, a free GPU):

    python3 tools/build_mla_dsl.py tools/kernels-bin

The DSL JIT is deterministic for a fixed toolchain; a different toolchain is
a different cubin and the manifests pin whichever one is checked in.
